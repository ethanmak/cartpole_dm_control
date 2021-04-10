from dm_control import suite
import matplotlib.pyplot as plt
from ddpg import DDPGAgent
from utils import *

NUM_EPISODES = 80
BATCH_SIZE = 128

env = suite.load(domain_name='cartpole', task_name='balance')  # type: Environment
action_spec = env.action_spec()

agent = DDPGAgent(state_dim=5, action_dim=1,
                  actor_learning_rate=1e-3, critic_learning_rate=1e-3,
                  gamma=0.99, tau=5e-3,
                  noise_std=0.2,
                  batch_size=BATCH_SIZE)

episode_rewards = []
average_rewards = []

for i in range(NUM_EPISODES):
    time_step = env.reset()
    prev_state = np.concatenate(list(time_step.observation.values()))
    episode_reward = 0
    count = 0
    while not time_step.last():
        action = np.clip(agent.get_noisy_action(prev_state),
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

plt.plot(average_rewards)
plt.show()

agent.save_models('test')