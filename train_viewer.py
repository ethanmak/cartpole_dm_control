from dm_control import suite
import numpy as np
from agents.ddpg import DDPGAgent
from dm_control import viewer

NUM_EPISODES = 120
BATCH_SIZE = 128

env = suite.load(domain_name='cartpole', task_name='swingup')  # type: Environment
action_spec = env.action_spec()
time_step = env.reset()
prev_state = np.concatenate(list(time_step.observation.values()))

agent = DDPGAgent(state_dim=prev_state.shape[0], action_dim=action_spec.shape[0],
                  actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                  gamma=0.99, tau=1e-2,
                  noise_std=0.2,
                  action_spec=action_spec,
                  batch_size=BATCH_SIZE)

prev_action = 0
episode_reward = 0

def train_policy(time_step):
    global episode_reward, prev_state
    current_state = np.concatenate(list(time_step.observation.values()))
    reward = time_step.reward
    if not time_step.first():
        agent.store_episode(prev_state, prev_action, reward, current_state)
    action = agent.get_policy(current_state)
    if reward is not None:
        episode_reward += reward
    if time_step.last():
        print('Reward: {}'.format(episode_reward))
        episode_reward = 0
    if agent.can_update():
        agent.update()
    prev_state = current_state
    return action

viewer.launch(env, policy=train_policy)
