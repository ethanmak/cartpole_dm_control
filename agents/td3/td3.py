import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import os
import numpy as np

from . import models
from baselines.ddpg.noise import *
from baselines.ddpg.memory import Memory
from ..common import constants

class TD3Agent:
    def __init__(self, state_dim, action_dim, actor_learning_rate, critic_learning_rate, gamma, tau,
                 action_spec, batch_size=128, policy_freq=2, epsilon=0, policy_noise=0.2, action_noise=0.2,
                 action_noise_clip=0.5, buffer_len=50000):
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
        self.policy_freq = policy_freq

        self.policy_noise = OrnsteinUhlenbeckActionNoise(np.zeros(1), policy_noise)
        self.policy_noise.reset()

        self.action_noise = action_noise
        self.action_noise_clip = action_noise_clip

        self.update_iter = tf.Variable(initial_value=0)

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
        action = self.get_action_tensor(state).numpy() + self.policy_noise() * self.action_range
        probability = np.random.binomial(1, self.epsilon, action.shape)
        return action + probability * (self._random_action() - action)

    @tf.function
    def get_action_tensor(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        return tf.squeeze(self.actor(state))

    def get_action(self, state):
        return np.clip(self.get_action_tensor(state).numpy(), self.action_min, self.action_max)

    def get_policy(self, state):
        return np.clip(self.get_action_tensor(state).numpy() + self.policy_noise() * self.action_range,
                       self.action_min, self.action_max)

    def can_update(self):
        return self.replay_buffer.nb_entries > self.batch_size

    def update(self):
        sample = self.replay_buffer.sample(self.batch_size)
        prev_state = tf.convert_to_tensor(sample['obs0'])
        state = tf.convert_to_tensor(sample['obs1'])
        reward = tf.cast(sample['rewards'], dtype=tf.float32)
        action = tf.convert_to_tensor(sample['actions'])

        noisy_action = action + np.clip(np.random.normal(loc=0.0, scale=self.action_noise, size=action.shape),
                                        -self.action_noise_clip, self.action_noise_clip) * self.action_range

        self._update_graph(prev_state, state, reward, noisy_action)

    @tf.function
    def _update_graph(self, prev_state, state, reward, action):
        with tf.GradientTape() as tape:
            target_action = self.actor_target(state, training=True)
            target_q_1, target_q_2 = self.critic_target([state, target_action], training=True)
            y = reward + self.gamma * tf.math.minimum(target_q_1, target_q_2)
            predicted_q_1, predicted_q_2 = self.critic([prev_state, action], training=True)
            critic_loss = self.loss_func(y, predicted_q_1) + self.loss_func(y, predicted_q_2)

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        self.update_iter.assign_add(1)

        if self.update_iter.value() % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                predicted_action = self.actor(prev_state, training=True)
                predicted_reward = self.critic([prev_state, predicted_action], training=True)
                actor_loss = -tf.reduce_mean(predicted_reward)

            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

            self.update_target(self.actor_target.variables, self.actor.variables, self.tau)
            self.update_target(self.critic_target.variables, self.critic.variables, self.tau)

    @tf.function
    def update_target(self, target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

    def save_models(self, model_name):
        path = os.path.join(constants.MODEL_SAVES, model_name)
        self.actor.save(os.path.join(path, 'actor'))
        self.critic.save(os.path.join(path, 'critic'))

    def load_models(self, model_name):
        path = os.path.join(constants.MODEL_SAVES, model_name)
        self.actor = keras.models.load_model(os.path.join(path, 'actor'))
        self.critic = keras.models.load_model(os.path.join(path, 'critic'))
