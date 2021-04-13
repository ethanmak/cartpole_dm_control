from tensorflow import keras
import tensorflow as tf
import os
import numpy as np

from . import constants


class SavedActor:
    def __init__(self, model_name, action_spec):
        path = os.path.join(constants.MODEL_SAVES, model_name)
        self.actor = keras.models.load_model(os.path.join(path, 'actor'))
        self.action_spec = action_spec

    @tf.function
    def get_action_tensor(self, state):
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        return tf.squeeze(self.actor(state))

    def get_action(self, state):
        return np.clip(self.get_action_tensor(state).numpy(), self.action_spec.minimum, self.action_spec.maximum)
