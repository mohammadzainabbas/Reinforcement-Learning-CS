from typing import Tuple
import dm_control
from dm_control.rl import control
import numpy as np
import tensorflow as tf

from sac import SACAgent
from utils import actor_model, critic_model, preprocess_data, print_log

def train_sac() -> Tuple[SACAgent, control.Environment, tf.keras.Model, tf.keras.Model]:
    # Defaults
    SEED = 42
    EPOCHS = 1000
    EPISODES = 10
    DISCOUNT_FACTOR = 0.99

    # Set the random seed for reproducibility.
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Load the `manipulator` environment.
    env = dm_control.suite.load('manipulator', 'insert_ball', visualize_reward=True)

    # Create the actor and critic models.
    actor = actor_model(env=env)
    critic = critic_model(env=env)

    # Create the SAC agent.
    agent = SACAgent(actor, critic)

    # Set the discount factor and target entropy.
    agent.discount_factor = DISCOUNT_FACTOR
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
        print_log(f'Restored from {manager.latest_checkpoint}')
    else:
        print_log('Initializing from scratch.')

    # Training loop.
    for epoch in range(EPOCHS):
        # Collect a few episodes of data.
        episodes = []
        for _ in range(EPISODES):
            episode = []
            time_step = env.reset()
            while not time_step.last():
                action = agent.select_action(time_step)
                next_time_step = env.step(action)
                episode.append((time_step, action, next_time_step))
                time_step = next_time_step
            episodes.append(episode)

        # Preprocess the data.
        observations, actions, rewards, next_observations, dones = preprocess_data(episodes)

        # Update the agent.
        agent.train_step(observations, actions, rewards, next_observations, dones, optimizer)

        # Save the checkpoint.
        manager.save()

    # Close the environment.
    env.close()

    return agent, env, actor, critic

if __name__ == "__main__":
    pass