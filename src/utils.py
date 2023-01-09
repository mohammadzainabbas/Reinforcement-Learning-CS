import tensorflow as tf
from keras.layers import Dense
from keras import Model
from dm_control import suite
# Define the actor model.
def actor_model():
    inputs = tf.keras.Input(shape=(env.observation_spec().shape[0],))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    logits = Dense(env.action_spec().shape[0])(x)
    model = Model(inputs=inputs, outputs=logits)
    return model
