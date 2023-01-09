import tensorflow as tf
from tensorflow.keras.layers import Dense

# Define the actor model.
def actor_model():
    inputs = tf.keras.Input(shape=(env.observation_spec().shape[0],))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    logits = Dense(env.action_spec().shape[0])(x)
    model = Model(inputs=inputs, outputs=logits)
    return model
