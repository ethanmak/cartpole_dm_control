from dm_control import suite
from dm_control import viewer
import numpy as np
import tensorflow as tf

NUM_EPISODES = 120
BATCH_SIZE = 128

env = suite.load(domain_name='cartpole', task_name='swingup')  # type: Environment
action_spec = env.action_spec()

time_step = env.reset()

print(tf.test.gpu_device_name())

def random_policy(time_step):
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    return action

while not time_step.last():
    action = random_policy(time_step)
    time_step = env.step(action)

viewer.launch(env, policy=random_policy)
