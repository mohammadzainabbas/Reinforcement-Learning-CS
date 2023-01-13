import functools
from datetime import datetime
import time
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt

from os import getcwd
from os.path import join

from absl import logging

from brax import envs
from brax import jumpy as jp
from brax.envs import wrappers
from brax.io import model
from brax.io import html
from brax.training import acting
from brax.training import gradients
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey

import flax
import jax
import jax.numpy as jnp
import optax

InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'

@flax.struct.dataclass
class TrainingState:
	"""Contains training state for the learner."""
	optimizer_state: optax.OptState
	params: ppo_losses.PPONetworkParams
	normalizer_params: running_statistics.RunningStatisticsState
	env_steps: jnp.ndarray

def _unpmap(v):
	return jax.tree_util.tree_map(lambda x: x[0], v)

def train(
	environment: envs.Env,
	num_timesteps: int,
	episode_length: int,
	action_repeat: int = 1,
	num_envs: int = 1,
	max_devices_per_host: Optional[int] = None,
	num_eval_envs: int = 128,
	learning_rate: float = 1e-4,
	entropy_cost: float = 1e-4,
	discounting: float = 0.9,
	seed: int = 0,
	unroll_length: int = 10,
	batch_size: int = 32,
	num_minibatches: int = 16,
	num_updates_per_batch: int = 2,
	num_evals: int = 1,
	normalize_observations: bool = False,
	reward_scaling: float = 1.,
	clipping_epsilon: float = .3,
	gae_lambda: float = .95,
	deterministic_eval: bool = False,
	network_factory: types.NetworkFactory[ppo_networks.PPONetworks] = ppo_networks.make_ppo_networks,
	progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
	normalize_advantage: bool = True,
	eval_env: Optional[envs.Env] = None):
	"""
	Proximal policy optimization training.

	See: https://arxiv.org/pdf/1707.06347.pdf
	"""
	assert batch_size * num_minibatches % num_envs == 0
	xt = time.time()

	process_count = jax.process_count()
	process_id = jax.process_index()
	local_device_count = jax.local_device_count()
	local_devices_to_use = local_device_count
	if max_devices_per_host:
		local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
	logging.info(
		'Device count: %d, process count: %d (id %d), local device count: %d, '
		'devices to be used count: %d', jax.device_count(), process_count,
		process_id, local_device_count, local_devices_to_use)
	device_count = local_devices_to_use * process_count

	# The number of environment steps executed for every training step.
	env_step_per_training_step = (batch_size * unroll_length * num_minibatches * action_repeat)
	num_evals_after_init = max(num_evals - 1, 1)
	# The number of training_step calls per training_epoch call.
	# equals to ceil(num_timesteps / (num_evals * env_step_per_training_step))
	num_training_steps_per_epoch = -(-num_timesteps // (num_evals_after_init * env_step_per_training_step))

	assert num_envs % device_count == 0
	env = environment

	env = wrappers.wrap_for_training(env, episode_length=episode_length, action_repeat=action_repeat)

	reset_fn = jax.jit(jax.vmap(env.reset))

	normalize = lambda x, y: x
	if normalize_observations:
		normalize = running_statistics.normalize
	ppo_network = network_factory(
		env.observation_size,
		env.action_size,
		preprocess_observations_fn=normalize)
	make_policy = ppo_networks.make_inference_fn(ppo_network)

	optimizer = optax.adam(learning_rate=learning_rate)

	loss_fn = functools.partial(
		ppo_losses.compute_ppo_loss,
		ppo_network=ppo_network,
		entropy_cost=entropy_cost,
		discounting=discounting,
		reward_scaling=reward_scaling,
		gae_lambda=gae_lambda,
		clipping_epsilon=clipping_epsilon,
		normalize_advantage=normalize_advantage)

	gradient_update_fn = gradients.gradient_update_fn(
		loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True)

	def minibatch_step(
		carry, data: types.Transition,
		normalizer_params: running_statistics.RunningStatisticsState):
		optimizer_state, params, key = carry
		key, key_loss = jax.random.split(key)
		(_, metrics), params, optimizer_state = gradient_update_fn(
			params,
			normalizer_params,
			data,
			key_loss,
			optimizer_state=optimizer_state)

		return (optimizer_state, params, key), metrics

	def sgd_step(carry, unused_t, data: types.Transition,
				normalizer_params: running_statistics.RunningStatisticsState):
		optimizer_state, params, key = carry
		key, key_perm, key_grad = jax.random.split(key, 3)

		def convert_data(x: jnp.ndarray):
			x = jax.random.permutation(key_perm, x)
			x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
			return x

		shuffled_data = jax.tree_util.tree_map(convert_data, data)
		(optimizer_state, params, _), metrics = jax.lax.scan(
			functools.partial(minibatch_step, normalizer_params=normalizer_params),
			(optimizer_state, params, key_grad),
			shuffled_data,
			length=num_minibatches)
		return (optimizer_state, params, key), metrics

	def training_step(
		carry: Tuple[TrainingState, envs.State, PRNGKey],
		unused_t) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
		training_state, state, key = carry
		key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

		policy = make_policy(
			(training_state.normalizer_params, training_state.params.policy))

		def f(carry, unused_t):
			current_state, current_key = carry
			current_key, next_key = jax.random.split(current_key)
			next_state, data = acting.generate_unroll(
				env,
				current_state,
				policy,
				current_key,
				unroll_length,
				extra_fields=('truncation',))
			return (next_state, next_key), data

		(state, _), data = jax.lax.scan(
			f, (state, key_generate_unroll), (),
			length=batch_size * num_minibatches // num_envs)
		# Have leading dimentions (batch_size * num_minibatches, unroll_length)
		data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
		data = jax.tree_util.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
									data)
		assert data.discount.shape[1:] == (unroll_length,)

		# Update normalization params and normalize observations.
		normalizer_params = running_statistics.update(
			training_state.normalizer_params,
			data.observation,
			pmap_axis_name=_PMAP_AXIS_NAME)

		(optimizer_state, params, _), metrics = jax.lax.scan(
			functools.partial(
				sgd_step, data=data, normalizer_params=normalizer_params),
			(training_state.optimizer_state, training_state.params, key_sgd), (),
			length=num_updates_per_batch)

		new_training_state = TrainingState(
			optimizer_state=optimizer_state,
			params=params,
			normalizer_params=normalizer_params,
			env_steps=training_state.env_steps + env_step_per_training_step)
		return (new_training_state, state, new_key), metrics

	def training_epoch(training_state: TrainingState, state: envs.State,
						key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
		(training_state, state, _), loss_metrics = jax.lax.scan(
			training_step, (training_state, state, key), (),
			length=num_training_steps_per_epoch)
		loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
		return training_state, state, loss_metrics

	training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME)

	# Note that this is NOT a pure jittable method.
	def training_epoch_with_timing(
		training_state: TrainingState, env_state: envs.State,
		key: PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
		nonlocal training_walltime
		t = time.time()
		(training_state, env_state,
		metrics) = training_epoch(training_state, env_state, key)
		metrics = jax.tree_util.tree_map(jnp.mean, metrics)
		jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

		epoch_training_time = time.time() - t
		training_walltime += epoch_training_time
		sps = (num_training_steps_per_epoch *
			env_step_per_training_step) / epoch_training_time
		metrics = {
			'training/sps': sps,
			'training/walltime': training_walltime,
			**{f'training/{name}': value for name, value in metrics.items()}
		}
		return training_state, env_state, metrics

	key = jax.random.PRNGKey(seed)
	global_key, local_key = jax.random.split(key)
	del key
	local_key = jax.random.fold_in(local_key, process_id)
	local_key, key_env, eval_key = jax.random.split(local_key, 3)
	# key_networks should be global, so that networks are initialized the same
	# way for different processes.
	key_policy, key_value = jax.random.split(global_key)
	del global_key

	init_params = ppo_losses.PPONetworkParams(
		policy=ppo_network.policy_network.init(key_policy),
		value=ppo_network.value_network.init(key_value))
	training_state = TrainingState(
		optimizer_state=optimizer.init(init_params),
		params=init_params,
		normalizer_params=running_statistics.init_state(
			specs.Array((env.observation_size,), jnp.float32)),
		env_steps=0)
	training_state = jax.device_put_replicated(
		training_state,
		jax.local_devices()[:local_devices_to_use])

	key_envs = jax.random.split(key_env, num_envs // process_count)
	key_envs = jnp.reshape(key_envs,
							(local_devices_to_use, -1) + key_envs.shape[1:])
	env_state = reset_fn(key_envs)

	if not eval_env:
		eval_env = env
	else:
		eval_env = wrappers.wrap_for_training(
			eval_env, episode_length=episode_length, action_repeat=action_repeat)

	evaluator = acting.Evaluator(
		eval_env,
		functools.partial(make_policy, deterministic=deterministic_eval),
		num_eval_envs=num_eval_envs,
		episode_length=episode_length,
		action_repeat=action_repeat,
		key=eval_key)

	# Run initial eval
	if process_id == 0 and num_evals > 1:
		metrics = evaluator.run_evaluation(
			_unpmap(
				(training_state.normalizer_params, training_state.params.policy)),
			training_metrics={})
		logging.info(metrics)
		progress_fn(0, metrics)

	training_walltime = 0
	current_step = 0
	for it in range(num_evals_after_init):
		logging.info('starting iteration %s %s', it, time.time() - xt)

		# optimization
		epoch_key, local_key = jax.random.split(local_key)
		epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
		(training_state, env_state,
		training_metrics) = training_epoch_with_timing(training_state, env_state,
														epoch_keys)
		current_step = int(_unpmap(training_state.env_steps))

		if process_id == 0:
			# Run evals.
			metrics = evaluator.run_evaluation(
				_unpmap(
					(training_state.normalizer_params, training_state.params.policy)),
				training_metrics)
			logging.info(metrics)
			progress_fn(current_step, metrics)

	total_steps = current_step
	assert total_steps >= num_timesteps

	# If there was no mistakes the training_state should still be identical on all
	# devices.
	pmap.assert_is_replicated(training_state)
	params = _unpmap(
		(training_state.normalizer_params, training_state.params.policy))
	logging.info('total steps: %s', total_steps)
	pmap.synchronize_hosts()
	return (make_policy, params, metrics)

def main() -> None:

	ENV_NAME, SEED = "grasp", 0
	env = envs.get_environment(env_name=ENV_NAME)
	state = env.reset(rng=jp.random_prngkey(seed=SEED))

	train_fn = functools.partial(
		train, 
		num_timesteps=600_000_000, 
		num_evals=10, 
		reward_scaling=10, 
		episode_length=1000, 
		normalize_observations=True, 
		action_repeat=1, 
		unroll_length=20, 
		num_minibatches=32, 
		num_updates_per_batch=2, 
		discounting=0.99, 
		learning_rate=3e-4, 
		entropy_cost=0.001, 
		num_envs=2048, 
		batch_size=256)

	max_y, min_y = 100, 0
	xdata, ydata = [], []

	times = [datetime.now()]

	def progress(num_steps, metrics):
		"""
		Callback function for training progress.
		"""
		times.append(datetime.now())
		xdata.append(num_steps)
		ydata.append(metrics['eval/episode_reward'])
	
	# Run training.
	make_inference_fn, params, metrics = train_fn(environment=env, progress_fn=progress)

	print(f'Time to jit: {times[1] - times[0]}')
	print(f'Time to train: {times[-1] - times[1]}')

	# Save plot
	_, ax = plt.subplots()

	plt.xlim([0, train_fn.keywords['num_timesteps']])
	plt.ylim([min_y, max_y])
	plt.xlabel('# environment steps')
	plt.ylabel('reward per episode')

	ax.plot(xdata, ydata, label="PPO")
	ax.legend()
	plt.savefig('ppo_training.png')

	# Save the trained model.
	output_model_path = join(getcwd(), "ppo_params")
	model.save_params(output_model_path, params)

	inference_fn = make_inference_fn(params)

	env = envs.create(env_name=ENV_NAME)
	jit_env_reset = jax.jit(env.reset)
	jit_env_step = jax.jit(env.step)
	jit_inference_fn = jax.jit(inference_fn)

	rollout = []
	rng = jax.random.PRNGKey(seed=SEED)
	state = jit_env_reset(rng=rng)
	for _ in range(1000):
		rollout.append(state)
		act_rng, rng = jax.random.split(rng)
		act, _ = jit_inference_fn(state.obs, act_rng)
		state = jit_env_step(state, act)

	html_result = html.render(env.sys, [s.qp for s in rollout])
	with open('result_with_ppo.html', 'w') as f:
		f.write(html_result)

if __name__ == '__main__':
	main()