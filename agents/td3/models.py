import tensorflow as tf
from tensorflow.keras import layers

def get_actor(state_dim, action_dim):
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(action_dim, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim))
    action_input = layers.Input(shape=(action_dim))

    state_out_1 = layers.Dense(16, activation="relu")(state_input)
    state_out_1 = layers.Dense(32, activation="relu")(state_out_1)

    action_out_1 = layers.Dense(32, activation="relu")(action_input)

    concat_1 = layers.Concatenate()([state_out_1, action_out_1])

    out_1 = layers.Dense(256, activation="relu")(concat_1)
    out_1 = layers.Dense(256, activation="relu")(out_1)
    output_1 = layers.Dense(1)(out_1)

    state_out_2 = layers.Dense(16, activation="relu")(state_input)
    state_out_2 = layers.Dense(32, activation="relu")(state_out_2)

    action_out_2 = layers.Dense(32, activation="relu")(action_input)

    concat_2 = layers.Concatenate()([state_out_2, action_out_2])

    out_2 = layers.Dense(256, activation="relu")(concat_2)
    out_2 = layers.Dense(256, activation="relu")(out_2)
    output_2 = layers.Dense(1)(out_2)

    model = tf.keras.Model([state_input, action_input], [output_1, output_2])

    return model
