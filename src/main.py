import dm_control
import dm_env
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from sac import SACAgent
from utils import actor_model, critic_model

# Set the random seed for reproducibility.
np.random.seed(42)
tf.random.set_seed(42)

# Load the `manipulator` environment.
env = dm_control.suite.load(
    'manipulator', 'insert_ball', task_kwargs=dict(target_size='large'))

# Create the actor and critic models.
actor = actor_model(env=env)
critic = critic_model(env=env)

# Create the SAC agent.
agent = SACAgent(actor, critic)

# Set the discount factor and target entropy.
agent.discount_factor = 0.99
agent.target_entropy = -np.prod(env.action_spec().shape)

# Set the learning rate and optimizer.
agent.learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate=agent.learning_rate)

# Create the checkpoint manager to save and restore the agent.
checkpoint_dir = './checkpoints'
checkpoint = tf.train.Checkpoint(optimizer=optimizer, agent=agent)
manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=3)

# Restore the checkpoint if it exists.
checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print(f'Restored from {manager.latest_checkpoint}')
else:
    print('Initializing from scratch.')

# Training loop.
for epoch in range(1000):
    # Collect a few episodes of data.
    episodes = []
    for _ in range(10):
        episode = []
        time_step = env.reset()
        while not time_step.last():
            action = agent.select_action(time_step)
            next_time_step = env.step(action)
            episode.append((time_step, action, next_time_step))
            time_step = next_time_step
        episodes.append(episode)

    # Preprocess the data.
    trajectories = rl_sac.preprocess_data(episodes)
    observations, actions, rewards, next_observations, dones = trajectories

    # Update the agent.
    agent.train_step(observations, actions, rewards, next_observations, dones, optimizer)

    # Save the checkpoint.
    manager.save()

# Close the environment.
env.close()