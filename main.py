from dm_control import suite
from dm_control import viewer
from agents.common.saved import SavedActor
import numpy as np

MODEL_NAME = 'td3_eps_greedy'

env = suite.load(domain_name='cartpole', task_name='swingup')  # type: Environment

episode_reward = 0

agent = SavedActor(model_name=MODEL_NAME, action_spec=env.action_spec())

def policy(time_step):
    global episode_reward
    if time_step.reward is not None:
        episode_reward += time_step.reward
    state = np.concatenate(list(time_step.observation.values()))
    return agent.get_action(state)


viewer.launch(env, policy=policy)

print('Episode Reward: {}'.format(episode_reward))
