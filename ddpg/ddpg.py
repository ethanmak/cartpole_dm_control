import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import optimizers
import os

import models
from baselines.ddpg.noise import *
from baselines.ddpg.memory import Memory

MODEL_SAVES = os.path.join(os.path.dirname(__file__), 'models')

class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_learning_rate, critic_learning_rate, gamma, tau, batch_size,
                 action_spec, epsilon=0, noise_std=0.2, buffer_len=50000):
        self.state_dim = state_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.action_dim = action_dim
        self.tau = tau
        self.batch_size = batch_size
        self.action_min = action_spec.minimum
        self.action_max = action_spec.maximum
        self.action_range = (action_spec.maximum - action_spec.minimum) / 2
        self.epsilon = epsilon

        self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(1), noise_std)
        self.noise.reset()

        buffer_shape = { 'state': state_dim,
                         'action': action_dim,
                         'reward': 1,
                         'next_state': state_dim,
                         'goal': state_dim}
        # self.replay_buffer = ReplayBuffer(buffer_shapes=buffer_shape)
        self.replay_buffer = Memory(buffer_len, (action_dim,), (state_dim,))

        self.actor = models.get_actor(state_dim, action_dim)
        self.actor_target = models.get_actor(state_dim, action_dim)
        self.critic = models.get_critic(state_dim, action_dim)
        self.critic_target = models.get_critic(state_dim, action_dim)

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(learning_rate=actor_learning_rate)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_learning_rate)

        self.loss_func = keras.losses.MeanSquaredError()

    def store_episode(self, prev_state, action, reward, state, training=True):
        self.replay_buffer.append(prev_state, action, reward, state, terminal1=0, training=training)

    def _random_action(self):
        return np.random.uniform(self.action_min, self.action_max, self.action_dim)

    def get_eps_greedy_policy(self, state):
        action = self.get_action_tensor(state).numpy() + self.noise() * self.action_range
        probability = np.random.binomial(1, self.epsilon, action.shape)
        return action + probability * (self._random_action() - action)

    @tf.function
    def get_action_tensor(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        return tf.squeeze(self.actor(state))

    def get_action(self, state):
        return np.clip(self.get_action_tensor(state).numpy(), self.action_min, self.action_max)

    def get_policy(self, state):
        return np.clip(self.get_action_tensor(state).numpy() + self.noise() * self.action_range,
                       self.action_min, self.action_max)

    def can_update(self):
        return self.replay_buffer.nb_entries > 2 * self.batch_size

    def update(self):
        sample = self.replay_buffer.sample(self.batch_size)
        prev_state = tf.convert_to_tensor(sample['obs0'])
        state = tf.convert_to_tensor(sample['obs1'])
        reward = tf.cast(sample['rewards'], dtype=tf.float32)
        action = tf.convert_to_tensor(sample['actions'])

        self._update_graph(prev_state, state, reward, action)

        self.update_target(self.actor_target.variables, self.actor.variables, self.tau)
        self.update_target(self.critic_target.variables, self.critic.variables, self.tau)

    @tf.function
    def _update_graph(self, prev_state, state, reward, action):
        with tf.GradientTape() as tape:
            target_action = self.actor_target(state, training=True)
            y = reward + self.gamma * self.critic_target([state, target_action], training=True)
            predicted_q = self.critic([prev_state, action], training=True)
            critic_loss = self.loss_func(y, predicted_q)

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            predicted_action = self.actor(prev_state, training=True)
            predicted_reward = self.critic([prev_state, predicted_action], training=True)
            actor_loss = -tf.reduce_mean(predicted_reward)

        # tf.print('Critic Loss:', critic_loss, 'Actor Loss:', actor_loss)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def save_models(self, model_name):
        path = os.path.join(MODEL_SAVES, model_name)
        self.actor.save(os.path.join(path, 'actor'))
        self.actor_target.save(os.path.join(path, 'actor_target'))
        self.critic.save(os.path.join(path, 'critic'))

    def load_models(self, model_name):
        path = os.path.join(MODEL_SAVES, model_name)
        self.actor = keras.models.load_model(os.path.join(path, 'actor'))
        self.actor_target = keras.models.load_model(os.path.join(path, 'actor_target'))
        self.critic = keras.models.load_model(os.path.join(path, 'critic'))
