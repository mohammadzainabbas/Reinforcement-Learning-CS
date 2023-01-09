import tensorflow as tf
from keras.layers import Dense
from keras import Model
from dm_control import suite

# Define the actor model.
def actor_model(env: suite.Environment):
    inputs = tf.keras.Input(shape=(env.observation_spec().shape[0],))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    logits = Dense(env.action_spec().shape[0])(x)
    model = Model(inputs=inputs, outputs=logits)
    return model

# Define the critic model.
def critic_model(env: suite.Environment):
    inputs = tf.keras.Input(shape=(env.observation_spec().shape[0],))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    value = Dense(1)(x)
    model = Model(inputs=inputs, outputs=value)
    return model

# Preprocess the data.
def preprocess_data(episodes):
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
