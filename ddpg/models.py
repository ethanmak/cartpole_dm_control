import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import initializers

class Actor(keras.Model):
    def __init__(self, state_dim, action_dim, hidden_layers_dim):
        """

        :param state_dim: int describing number of dims state
        :param action_dim:
        :param hidden_layers_dim: list of ints
        """
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_layer = layers.InputLayer(input_shape=(None, state_dim))
        self.hidden_layers = []
        self.output_layer = layers.Dense(action_dim, activation='tanh',
                                   kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   bias_initializer=keras.initializers.RandomUniform(minval=-3e-4, maxval=3e-4))
        for l in hidden_layers_dim:
            self.hidden_layers.append(layers.Dense(l))
            self.hidden_layers.append(layers.BatchNormalization())
            self.hidden_layers.append(layers.Activation(activation=activations.relu))

    def call(self, inputs, training=None, mask=None):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class Critic(keras.Model):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.q_dim = q_dim
        self.state_input = layers.InputLayer(input_shape=(None, state_dim))
        self.action_input = layers.InputLayer(input_shape=(None, action_dim))
        self.state_layers = [layers.Dense(400),
                             layers.BatchNormalization(),
                             layers.Activation(activation=activations.relu),
                             layers.Dense(300, use_bias=False)]
        self.action_layers = [layers.Dense(300)]
        self.concat = layers.Concatenate()
        self.combined_layers = [layers.Activation(activation=activations.relu)]
        self.output_layer = layers.Dense(1, kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   bias_initializer=keras.initializers.RandomUniform(minval=-3e-4, maxval=3e-4))

    def call(self, inputs, training=None, mask=None):
        state = self.state_input(inputs[0])
        action = self.action_input(inputs[1])
        for layer in self.state_layers:
            state = layer(state)
        for layer in self.action_layers:
            action = layer(action)
        combined = self.concat([state, action])
        for layer in self.combined_layers:
            combined = layer(combined)
        return self.output_layer(combined)


def get_actor_model(state_dim, action_dim):
    state_input = keras.Input(shape=(state_dim,))
    state = layers.Dense(400)(state_input)
    state = layers.BatchNormalization()(state)
    state = layers.Activation(activation=activations.relu)(state)
    state = layers.Dense(300)(state)
    state = layers.BatchNormalization()(state)
    state = layers.Activation(activation=activations.relu)(state)

    output = layers.Dense(action_dim, activation='tanh',
                                   kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                                   bias_initializer=keras.initializers.RandomUniform(minval=-3e-4, maxval=3e-4))(state)
    return keras.Model(state_input, output)


def get_critic_model(state_dim, action_dim):
    state_input = keras.Input(shape=(state_dim,))
    action_input = keras.Input(shape=(action_dim,))

    state = layers.Dense(400)(state_input)
    state = layers.BatchNormalization()(state)
    state = layers.Activation(activation=activations.relu)(state)
    state = layers.Dense(300, use_bias=False)(state)

    action = layers.Dense(300)(action_input)
    output = layers.Concatenate()([state, action])
    output = layers.Activation(activation=activations.relu)(output)
    output = layers.Dense(1, kernel_initializer=initializers.RandomUniform(minval=-3e-3, maxval=3e-3),
                          bias_initializer=keras.initializers.RandomUniform(minval=-3e-4, maxval=3e-4))(output)

    return keras.Model([state_input, action_input], output)

def get_actor(state_dim, action_dim):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(action_dim, activation="tanh", kernel_initializer=last_init)(out)

    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(state_dim, action_dim):
    # State as input
    state_input = layers.Input(shape=(state_dim))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(action_dim))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model
