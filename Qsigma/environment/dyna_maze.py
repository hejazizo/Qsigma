import logging

import numpy as np
from Qsigma.environment.env import Environment

logger = logging.getLogger(__name__)

# World
WIDTH = None
STATE2WORLD = []
STATES = []
POSITION = {}

# Velocity
V = None
V_MIN = None
V_MAX = None

# Actions
ACTIONS = range(9)

_RIGHT = 0
RIGHT = 1
RIGHT_ = 2

_UP = 3
UP = 4
UP_ = 5

_LEFT = 6
LEFT = 7
LEFT_ = 8

# Rewards
REWARD = {}


class DynaMaze(Environment):
    def __init__(self, config: str):
        super().__init__(config)

    def act(self, s, a):
        """
        Perform action a in state s, and observe r in s'
        :param s: current state
        :param a: action to take from state s
        :param beta: proba of no velocity update (environment stochasticity)
        :return: next state and observed reward
        """

        self.name = 'DynaMaze'
        # update velocity with probability 1-beta
        global V, V_MIN, V_MAX, REWARD, POSITION
        if np.random.random() < 1-self.config['beta']:
            if a in [_RIGHT, _UP, _LEFT] and V > V_MIN:
                V -= 1
            elif a in [RIGHT_, UP_, LEFT_] and V < V_MAX:
                V += 1

        r_border = range(WIDTH-1, WIDTH**2, WIDTH)  # states on the right border
        l_border = range(0, WIDTH**2, WIDTH)  # states on the left border
        t_border = range(WIDTH)  # states on the top border

        units = range(V)
        check = False  # flag to indicate if we visited the checkpoint

        # ----------------------------- #
        # --- move RIGHT of V units --- #
        # ----------------------------- #
        if a < len(ACTIONS) / 3:
            for i in units:
                self.world[STATE2WORLD[s+i]] = '>'  # draw my path gradualy in the world
                # crash: reset world and velocities, return to start state
                if s+i in r_border or s+i+1 in POSITION['WALLS']:
                    self.reset()
                    return POSITION['START'], REWARD['CRASH'], False
                # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
                elif s+i+1 == POSITION['CHECKPNT']:
                    check = V_MAX != 5
                    V_MAX = 5
                # goal: draw where I end up & return
                elif s+i+1 == POSITION['GOAL']:
                    self.world[STATE2WORLD[s+i+1]] = 'O'
                    return s+i+1, REWARD['WIN'], True
            # draw where I end up & return
            self.world[STATE2WORLD[s+V]] = 'O'
            return (s+V, REWARD['CHECKPNT'], False) if check else (s+V, REWARD['STEP'], False)

        # ----------------------------- #
        # ---- move UP of V units ----- #
        # ----------------------------- #
        elif a < 2*len(ACTIONS) / 3:
            for i in units:
                self.world[STATE2WORLD[s-i*WIDTH]] = '|'  # draw my path gradualy in the world
                # crash: reset world and velocities, return to start state
                if s-i*WIDTH in t_border or s-(i+1)*WIDTH in POSITION['WALLS']:
                    self.reset()
                    return POSITION['START'], REWARD['CRASH'], False
                # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
                elif s-(i+1)*WIDTH == POSITION['CHECKPNT']:
                    check = V_MAX != 5
                    V_MAX = 5
                # goal: draw where I end up & return
                elif s-(i+1)*WIDTH == POSITION['GOAL']:
                    self.world[STATE2WORLD[s-(i+1)*WIDTH]] = 'O'
                    return s-(i+1)*WIDTH, REWARD['WIN'], True
            # nothing special: draw where I end up & return
            self.world[STATE2WORLD[s-V*WIDTH]] = 'O'
            return (s-V*WIDTH, REWARD['CHECKPNT'], False) if check else (s-V*WIDTH, REWARD['STEP'], False)

        # ----------------------------- #
        # --- move LEFT of V units ---- #
        # ----------------------------- #
        elif a < len(ACTIONS):
            for i in units:
                self.world[STATE2WORLD[s-i]] = '<'  # draw my path gradualy in the world
                # crash: reset world and velocities, return to start state
                if s-i in l_border or s-i-1 in POSITION['WALLS']:
                    self.reset()
                    return POSITION['START'], REWARD['CRASH'], False
                # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
                elif s-i-1 == POSITION['CHECKPNT']:
                    check = V_MAX != 5
                    V_MAX = 5
                # goal: draw where I end up & return
                elif s-i-1 == POSITION['GOAL']:
                    self.world[STATE2WORLD[s-i-1]] = 'O'
                    return s-i-1, REWARD['WIN'], True
            # draw where I end up & return
            self.world[STATE2WORLD[s-V]] = 'O'
            return (s-V, REWARD['CHECKPNT'], False) if check else (s-V, REWARD['STEP'], False)

        return s, REWARD['STEP'], False  # WARNING: SHOULD NEVER HAPPEN

    def reset(self):
        """
        reset grid world and velocities
        """

        # reset WORLD
        global STATES, WIDTH, STATE2WORLD
        self.world = np.array([
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "G"],
            ["_", "X", "X", "X", "X", "X", "X", "X", "X", "X"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["X", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "#", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "_"],
            ["S", "_", "_", "_", "_", "_", "_", "_", "_", "_"]
        ])

        STATES = range(self.world.size)
        self.width, self.height = self.world.shape
        WIDTH = self.width

        grid = np.indices((self.width, self.height))
        STATE2WORLD = [(x, y) for x, y in zip(grid[0].flatten(), grid[1].flatten())]

        # reset VELOCITY
        global V, V_MIN, V_MAX
        V = 1
        V_MAX = self.config['V_MAX']
        V_MIN = self.config['V_MIN']

        # initial objects
        global POSITION
        POSITION = self.config['position']

        # rewards
        global REWARD
        REWARD = self.config['reward']

    @property
    def position(self):
        global POSITION
        return POSITION

    @property
    def states(self):
        global STATES
        return STATES

    @property
    def actions(self):
        global ACTIONS
        return ACTIONS

    @property
    def STATE2WORLD(self):
        return STATE2WORLD

    @property
    def MOVEMENT2ARROW(self):
        return {0: u"\u2192", 1: u"\u2192", 2: u"\u2192",
                3: u"\u2191", 4: u"\u2191", 5: u"\u2191",
                6: u"\u2190", 7: u"\u2190", 8: u"\u2190"}
