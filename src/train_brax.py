import dm_control
import brax

# Load the environment.
env = dm_control.suite.load('manipulator', 'insert_ball', visualize_reward=True)

# Convert the environment to a gym.Env.
gym_env = brax.dm_control_to_gym_wrapper(env)

# Train the agent.
agent = brax.sac(gym_env,
                 replay_buffer_size=10000,
                 num_episodes=1000,
                 num_steps_per_epoch=1000,
                 num_steps_per_eval=1000,
                 epochs_per_eval=5,
                 max_steps=5000000)

# Test the agent.
average_reward, std_dev = brax.evaluate(gym_env, agent, num_episodes=100)
print(f'Average reward: {average_reward:.2f} +/- {std_dev:.2f}')

# Animate the agent.
brax.animate(gym_env, agent)