from dm_control import suite
from dm_control import viewer
import numpy as np
import tensorflow as tf
from dm_env import TimeStep

NUM_EPISODES = 120
BATCH_SIZE = 128

env = suite.load(domain_name='acrobot', task_name='swingup')  # type: Environment
action_spec = env.action_spec()

time_step = env.reset()
episode_reward = 0

print('GPU:', tf.test.gpu_device_name())
print('ActionSpec:', action_spec)

def random_policy(time_step: TimeStep):
    global episode_reward
    action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
    if time_step.first():
        episode_reward = 0
    else:
        episode_reward += time_step.reward
    print('Action:', action)
    print('Observation:', time_step.observation)
    print('Reward:', time_step.reward)
    if time_step.last():
        print('Episode Reward:', episode_reward)
    print()
    return action

viewer.launch(env, policy=random_policy)

print('Episode Reward:', episode_reward)
