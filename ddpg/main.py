from dm_control import suite
from dm_control import viewer
from ddpg import DDPGAgent
from utils import *

MODEL_NAME = 'ddpg_eps_greedy'

env = suite.load(domain_name='cartpole', task_name='swingup')  # type: Environment
agent = DDPGAgent(state_dim=5, action_dim=1,
                  actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                  gamma=0.99, tau=1e-2,
                  action_spec=env.action_spec(),
                  batch_size=128)

agent.load_models(MODEL_NAME)

episode_reward = 0


def policy(time_step):
    global episode_reward
    if time_step.reward is not None:
        episode_reward += time_step.reward
    state = np.concatenate(list(time_step.observation.values()))
    return agent.get_action(state)


viewer.launch(env, policy=policy)

print('Episode Reward: {}'.format(episode_reward))