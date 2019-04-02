import logging

import numpy as np
from rasa_nlu import utils
import math

logger = logging.getLogger(__name__)

# World
WORLD = None
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


class GridWorld():
    def __init__(self, config: str):
        self.config = config
        self.reset()

    def move(self, s, a):
        """
        Perform action a in state s, and observe r in s'
        :param s: current state
        :param a: action to take from state s
        :param beta: proba of no velocity update (environment stochasticity)
        :return: next state and observed reward
        """
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
                WORLD[STATE2WORLD[s+i]] = '>'  # draw my path gradualy in the world
                # crash: reset world and velocities, return to start state
                if s+i in r_border or s+i+1 in POSITION['WALLS']:
                    self.reset()
                    return POSITION['START'], REWARD['CRASH']
                # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
                elif s+i+1 == POSITION['CHECKPNT']:
                    check = V_MAX != 5
                    V_MAX = 5
                # goal: draw where I end up & return
                elif s+i+1 == POSITION['GOAL']:
                    WORLD[STATE2WORLD[s+i+1]] = 'O'
                    return s+i+1, REWARD['WIN']
            # draw where I end up & return
            WORLD[STATE2WORLD[s+V]] = 'O'
            return (s+V, REWARD['CHECKPNT']) if check else (s+V, REWARD['STEP'])

        # ----------------------------- #
        # ---- move UP of V units ----- #
        # ----------------------------- #
        elif a < 2*len(ACTIONS) / 3:
            for i in units:
                WORLD[STATE2WORLD[s-i*WIDTH]] = '|'  # draw my path gradualy in the world
                # crash: reset world and velocities, return to start state
                if s-i*WIDTH in t_border or s-(i+1)*WIDTH in POSITION['WALLS']:
                    self.reset()
                    return POSITION['START'], REWARD['CRASH']
                # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
                elif s-(i+1)*WIDTH == POSITION['CHECKPNT']:
                    check = V_MAX != 5
                    V_MAX = 5
                # goal: draw where I end up & return
                elif s-(i+1)*WIDTH == POSITION['GOAL']:
                    WORLD[STATE2WORLD[s-(i+1)*WIDTH]] = 'O'
                    return s-(i+1)*WIDTH, REWARD['WIN']
            # nothing special: draw where I end up & return
            WORLD[STATE2WORLD[s-V*WIDTH]] = 'O'
            return (s-V*WIDTH, REWARD['CHECKPNT']) if check else (s-V*WIDTH, REWARD['STEP'])

        # ----------------------------- #
        # --- move LEFT of V units ---- #
        # ----------------------------- #
        elif a < len(ACTIONS):
            for i in units:
                WORLD[STATE2WORLD[s-i]] = '<'  # draw my path gradualy in the world
                # crash: reset world and velocities, return to start state
                if s-i in l_border or s-i-1 in POSITION['WALLS']:
                    self.reset()
                    return POSITION['START'], REWARD['CRASH']
                # went through the checkpoint: increase V_MAX and return bonus (only the first time!)
                elif s-i-1 == POSITION['CHECKPNT']:
                    check = V_MAX != 5
                    V_MAX = 5
                # goal: draw where I end up & return
                elif s-i-1 == POSITION['GOAL']:
                    WORLD[STATE2WORLD[s-i-1]] = 'O'
                    return s-i-1, REWARD['WIN']
            # draw where I end up & return
            WORLD[STATE2WORLD[s-V]] = 'O'
            return (s-V, REWARD['CHECKPNT']) if check else (s-V, REWARD['STEP'])

        return s, REWARD['STEP']  # WARNING: SHOULD NEVER HAPPEN

    def choose_sigma(self, method, q_mode=None, last_sigma=None, base=None, t=None):
        """
        Return a sigma for the Q(sigma) algorithm according to a given method
        :param method: the algorithm to follow: SARSA, or TreeBackup, or Qsigma
        :param q_mode: Qsigma mode (only if method='Qsigma'): random, alternative, decreasing, or increasing mode
        :param last_sigma: the previous sigma returned by this function (only used for Qsigma in alternating mode)
        :param base: base of the logarithm to take (only used in non-alternating mode)
        :param t: current time step (only used for Qsigma in non-alternating mode)
        :return: 1 for SARSA, 0 for TreeBackup, 1 with probability p for Qsigma (in non-alternating mode)
        """

        sigma = {
            "SARSA": 1,
            "TreeBackup": 0,
            "Qsigma": {
                "rnd": None,    # RANDOM mode
                "alt": None,    # ALTERNATING mode
                "inc": None,    # INCREASING probability mode
                "dec": None     # DECREASING probability mode
            }
        }

        if method == "Qsigma":
            if q_mode == "rnd":
                sigma[method]["rnd"] = 1 if np.random.random() < 0.5 else 0
            elif q_mode == "alt":
                assert last_sigma in [0, 1]
                sigma[method]["rnd"] = 1 - last_sigma
            elif q_mode == "inc":
                assert base >= 3
                assert t >= 0
                sample_proba = 1 - math.exp(-math.log(1+t, base))  # increases with t
                sigma[method]["inc"] = 1 if np.random.random() < sample_proba else 0
            elif q_mode == "dec":
                assert base >= 3
                assert t >= 0
                sample_proba = math.exp(-math.log(1+t, base))  # decreases with t
                sigma[method]["dec"] = 1 if np.random.random() < sample_proba else 0

            return sigma[method][q_mode]

        return sigma[method]

    def reset(self):
        """
        reset grid world and velocities
        """

        # reset WORLD
        global WORLD, STATES, WIDTH, STATE2WORLD
        WORLD = np.array([
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

        STATES = range(WORLD.size)
        WIDTH = WORLD.shape[0]
        grid = np.indices((WIDTH, WIDTH))
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
    def map(self):
        global WORLD
        return WORLD


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='')
    utils.add_logging_option_arguments(parser)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    utils.configure_colored_logging(args.loglevel)

    grid_world = GridWorld()
    logger.info(f'\n{WORLD}')
    state, reward = grid_world.move(s=POSITION['START'], a=2, beta=0)
    logger.info(f'State = {state}, Reward = {reward}')
    logger.info(f'Positions: {grid_world.position}')


if __name__ == '__main__':
    main()
