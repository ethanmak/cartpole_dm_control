import numpy as np

def batchify(state):
    return state.reshape((1, -1))

def clip_actions(action, action_spec):
    return np.clip(action, action_spec.minimum, action_spec.maximum)