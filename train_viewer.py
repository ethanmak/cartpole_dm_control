from dm_control import suite
import numpy as np
from dm_control import viewer
from agents.td3 import TD3Agent
from agents.utils import stateify

NUM_EPISODES = 120
BATCH_SIZE = 128

env = suite.load(domain_name='cartpole', task_name='balance')  # type: Environment
action_spec = env.action_spec()
time_step = env.reset()
prev_state = stateify(time_step.observation)
action_range = action_spec.maximum - action_spec.minimum

agent = TD3Agent(state_dim=prev_state.shape[0], action_dim=action_spec.shape[0],
                 actor_learning_rate=5e-3, critic_learning_rate=7e-3,
                 gamma=0.99, tau=5e-3,
                 policy_noise=0.2 * action_range, action_noise=0.05, action_noise_clip=0.15,
                 action_spec=action_spec, epsilon=0.05, batch_size=BATCH_SIZE)

prev_action = 0
episode_reward = 0

def train_policy(time_step):
    global episode_reward, prev_state
    current_state = stateify(time_step.observation)
    reward = time_step.reward
    if not time_step.first():
        agent.store_episode(prev_state, prev_action, reward, current_state)
    action = agent.get_eps_greedy_policy(current_state)
    if reward is not None:
        episode_reward += reward
    if agent.can_update():
        agent.update()
    prev_state = current_state
    return action

viewer.launch(env, policy=train_policy)
