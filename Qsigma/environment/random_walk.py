import numpy as np
from Qsigma.environment.env import Environment

# true state value from bellman equation
TRUE_VALUE = np.arange(-20, 22, 2) / 20.0
TRUE_VALUE[0] = TRUE_VALUE[-1] = 0


class RandomWalk(Environment):
    def __init__(self, config: str):
        super().__init__(config)
        self.actions = [0, 1]
        self.states = range(self.config['n_states'])
        self.name = 'RandomWalk'

    def act(self, s, a):

        # next state
        if a == 0:
            next_state = s + 1
            self.world[next_state] = '>'
        elif a == 1:
            next_state = s - 1
            self.world[next_state] = '<'

        # reward
        if next_state == self.config['position']['END']['LEFT']:
            reward = self.config['reward']['END']['LEFT']
        elif next_state == self.config['position']['END']['RIGHT']:
            reward = self.config['reward']['END']['RIGHT']
        else:
            reward = self.config['reward']['STEP']

        # episode finished
        episode_finished = False
        if next_state in list(self.config['position']['END'].values()):
            episode_finished = True

        return next_state, reward, episode_finished

    def reset(self):
        self.world = ['__']*self.config['n_states']
        self.world[self.config['position']['START']] = 'S'
        self.world[self.config['position']['END']['RIGHT']] = 'X'
        self.world[self.config['position']['END']['LEFT']] = 'G'

    @property
    def position(self):
        return self.config['position']

    @property
    def MOVEMENT2ARROW(self):
        return {0: u"\u2192", 1: u"\u2190"}
