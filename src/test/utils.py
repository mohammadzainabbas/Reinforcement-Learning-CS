import tensorflow as tf
from keras.layers import Dense
from keras import Model
from dm_control.rl import control
import dm_env
import numpy as np
from typing import List, Tuple

def print_log(text: str, **kwargs): print(f"[ log ] {text}",**kwargs)
def print_error(text: str, **kwargs): print(f"[ error ] {text}",**kwargs)

# Define the actor model.
def actor_model(env: control.Environment):
    inputs = tf.keras.Input(shape=(env.observation_spec().shape[0],))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    logits = Dense(env.action_spec().shape[0])(x)
    model = Model(inputs=inputs, outputs=logits)
    return model

# Define the critic model.
def critic_model(env: control.Environment):
    inputs = tf.keras.Input(shape=(env.observation_spec().shape[0],))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    value = Dense(1)(x)
    model = Model(inputs=inputs, outputs=value)
    return model

# Preprocess the data.
def preprocess_data(episodes: List[List[Tuple[dm_env.TimeStep, np.ndarray, dm_env.TimeStep]]]):
    """
    This function takes a list of episodes, where each episode is a list of time steps. 
    It iterates over the episodes and time steps, and extracts the observations, actions, rewards, and next observations for each time step. 
    It then returns these data as NumPy arrays, which can be used to train the SAC agent.
    """
    observations, actions, rewards, next_observations, dones = [], [], [], [], []
    for episode in episodes:
        for time_step, action, next_time_step in episode:
            observations.append(time_step.observation)
            actions.append(action)
            rewards.append(next_time_step.reward)
            next_observations.append(next_time_step.observation)
            dones.append(next_time_step.last())
    return (
        np.array(observations),
        np.array(actions),
        np.array(rewards),
        np.array(next_observations),
        np.array(dones),
    )
