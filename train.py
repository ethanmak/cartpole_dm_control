from dm_control import suite
import matplotlib.pyplot as plt
from agents.ddpg import DDPGAgent
from agents.td3 import TD3Agent
import numpy as np
import os

MODEL_NAME = 'td3_eps_greedy'
NUM_EPISODES = 100
BATCH_SIZE = 128

env = suite.load(domain_name='cartpole', task_name='swingup')  # type: Environment
action_spec = env.action_spec()

# agent = DDPGAgent(state_dim=5, action_dim=action_spec[0],
#                   actor_learning_rate=1e-3, critic_learning_rate=1.5e-3,
#                   gamma=0.99, tau=5e-3,
#                   action_spec=action_spec,
#                   epsilon=0.05,
#                   batch_size=BATCH_SIZE)

agent = TD3Agent(state_dim=5, action_dim=action_spec.shape[0],
                 actor_learning_rate=1e-3, critic_learning_rate=1.5e-3,
                 gamma=0.99, tau=5e-3,
                 policy_noise=0.2, action_noise=0.05, action_noise_clip=0.15,
                 action_spec=action_spec, epsilon=0.05, batch_size=BATCH_SIZE)

episode_rewards = []
average_rewards = []

for i in range(NUM_EPISODES):
    time_step = env.reset()
    prev_state = np.concatenate(list(time_step.observation.values()))
    episode_reward = 0
    count = 0
    while not time_step.last():
        action = np.clip(agent.get_eps_greedy_policy(prev_state),
                         a_min=action_spec.minimum, a_max=action_spec.maximum)
        time_step = env.step(action)
        current_state = np.concatenate(list(time_step.observation.values()))
        reward = time_step.reward
        episode_reward += reward

        agent.store_episode(prev_state, action, reward, current_state)

        if agent.can_update():
            agent.update()
        prev_state = current_state
    episode_rewards.append(episode_reward)
    average_reward = np.mean(episode_rewards[-20:])
    average_rewards.append(average_reward)

    print('Episode {}, Reward: {}, Average Reward: {}'.format(i+1, episode_reward, average_reward))

figure_path = os.path.join(os.path.dirname(__file__), 'models', MODEL_NAME, 'reward.png')

agent.save_models(MODEL_NAME)

plt.title(MODEL_NAME + ' Average Reward')
plt.plot(average_rewards)
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.savefig(figure_path, format='png')
plt.show()