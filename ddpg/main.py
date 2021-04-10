from dm_control import suite
from dm_control import viewer
from ddpg import DDPGAgent
from utils import *

cartpole_env = suite.load(domain_name='cartpole', task_name='balance')  # type: Environment
agent = DDPGAgent(state_dim=5, action_dim=1,
                  actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                  gamma=0.99, tau=1e-2, noise_std=0.2,
                  batch_size=128)

agent.load_models('test')

episode_reward = 0

def policy(time_step):
    global episode_reward
    if time_step.reward is not None:
        episode_reward += time_step.reward
    state = np.concatenate(list(time_step.observation.values()))
    return agent.get_action(state)

viewer.launch(cartpole_env, policy=policy)
print('Episode Reward: {}'.format(episode_reward))