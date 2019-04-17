import numpy as np
from Qsigma.environment.env import Environment

STATE2WORLD = []


class WindyGridWorld(Environment):
    def __init__(self, config: str):
        super().__init__(config)
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7]     # UP, DOWN, LEFT, RIGHT, UP_RIGHT, UP_LEFT, DOWN_RIGHT, DOWN_LEFT
        self.wind = config['wind']
        self.name = 'WindyGridWorld'

    def act(self, s, a):

        i, j = STATE2WORLD[s]

        # next state
        if a == 0:
            next_state = [max(i - 1 - self.wind[j], 0), j]
            self.world[next_state[0], next_state[1]] = '^'
        elif a == 1:
            next_state = [max(min(i + 1 - self.wind[j], self.height - 1), 0), j]
            self.world[next_state[0], next_state[1]] = 'v'
        elif a == 2:
            next_state = [max(i - self.wind[j], 0), max(j - 1, 0)]
            self.world[next_state[0], next_state[1]] = '<'
        elif a == 3:
            next_state = [max(i - self.wind[j], 0), min(j + 1, self.width - 1)]
            self.world[next_state[0], next_state[1]] = '>'
        elif a == 4:
            next_state = [max(i - 1 - self.wind[j], 0), min(j + 1, self.width - 1)]
            self.world[next_state[0], next_state[1]] = '/'
        elif a == 5:
            next_state = [max(i - 1 - self.wind[j], 0), max(j - 1, 0)]
            self.world[next_state[0], next_state[1]] = '\\'
        elif a == 6:
            next_state = [max(min(i + 1 - self.wind[j], self.height - 1), 0), min(j + 1, self.width - 1)]
            self.world[next_state[0], next_state[1]] = '/'
        elif a == 7:
            next_state = [max(min(i + 1 - self.wind[j], self.height - 1), 0), max(j - 1, 0)]
            self.world[next_state[0], next_state[1]] = '\\'
        elif a == 8:
            next_state = [max(i - self.wind[j], 0), j]
            self.world[next_state[0], next_state[1]] = '__'
        else:
            assert False

        # episode finished
        next_state = next_state[0]*self.width + next_state[1]
        episode_done = False
        if next_state == self.config['position']['GOAL']:
            episode_done = True

        return next_state, -1, episode_done

    def reset(self):
        self.world = np.array([
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["S", "_", "_", "_", "_", "_", "_", "G", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
        ])

        global STATE2WORLD
        self.states = range(self.world.size)
        self.height, self.width = self.world.shape
        grid = np.indices((self.height, self.width))
        STATE2WORLD = [(x, y) for x, y in zip(grid[0].flatten(), grid[1].flatten())]

    @property
    def position(self):
        return self.config['position']

    @property
    def STATE2WORLD(self):
        return STATE2WORLD

    @property
    def MOVEMENT2ARROW(self):
        return {0: u"\u2191", 1: u"\u2193", 2: u"\u2190", 3: u"\u2192",
                4: u"\u2197", 5:  u"\u2196", 6: u"\u2198", 7: u"\u2199",
                8: u"\u21b7"}
