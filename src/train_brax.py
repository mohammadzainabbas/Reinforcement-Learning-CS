import dm_control
import brax

# Load the environment.
env = dm_control.mujoco.wrapper.load('manipulator', 'insert_ball', target_size=0.1)

# Convert the environment to a gym.Env.
gym_env = brax.dm_control_to_gym_wrapper(env)

