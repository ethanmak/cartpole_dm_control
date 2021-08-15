from dm_control import suite
from dm_control import viewer
from agents.common.saved import SavedActor
import os
from agents.utils import stateify
from tkinter import Tk, filedialog
from agents.common import constants

if __name__ == '__main__':
    root = Tk()
    root.withdraw()

    MODEL_PATH = filedialog.askdirectory(title='Model Name', initialdir=constants.MODEL_SAVES)
    if len(MODEL_PATH) <= 0:
        exit()
    FULL_MODEL_NAME = os.path.basename(MODEL_PATH)

    MODEL_NAME, DOMAIN_NAME, TASK_NAME = FULL_MODEL_NAME.split('-')

    env = suite.load(domain_name=DOMAIN_NAME, task_name=TASK_NAME)  # type: Environment

    episode_reward = 0

    agent = SavedActor(model_name=FULL_MODEL_NAME, action_spec=env.action_spec())

    def policy(time_step):
        global episode_reward
        if time_step.reward is not None:
            episode_reward += time_step.reward
        state = stateify(time_step.observation)
        return agent.get_action(state)

    viewer.launch(env, policy=policy, title=FULL_MODEL_NAME)

    print('Episode Reward: {}'.format(episode_reward))
