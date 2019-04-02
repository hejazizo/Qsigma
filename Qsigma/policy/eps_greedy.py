import logging

import numpy as np
from rasa_nlu import utils

from Qsigma.environment.grid_world import ACTIONS

logger = logging.getLogger(__name__)


class EpsGreedy():
    def __init__(self, config: str):
        self.epsilon = config['epsilon']

    def make_greedy(self, state, PI, Q):
        """
        Make PI(s,:) greedy according to Q(s,:) for all actions for a given state s
        :param s: given state
        :param epsilon: probability of choosing a non-optimal action
        """
        # action probabilities = epsilon / (|A|-1) for all actions by default
        # over |A|-1 because 1 of them will be optimal and have probability 1-epsilon
        PI[state, :] = [self.epsilon / (len(ACTIONS) - 1.0)] * len(ACTIONS)

        # Get the best action for that state (greedy w.r.t. Q):
        best_action = np.argmax(Q[state, :])

        # Change default probability of best action to be 1-epsilon
        PI[state, best_action] = 1.0 - self.epsilon

        # Assert if probabilities sum to 1
        assert np.isclose(np.sum(PI[state, :]), 1.0)

    def choose_action(self, state, PI, Q):
        """
        Choose an action from state s according to epsilon-greedy policy
        :param s: current state
        :param epsilon: probability of choosing a non-optimal action
        :return: action to take from s
        """
        self.make_greedy(state, PI, Q)

        # sample from ACTIONS with proba distribution PI[s, :]
        return np.random.choice(ACTIONS, p=PI[state, :])


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='')
    utils.add_logging_option_arguments(parser)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    utils.configure_colored_logging(args.loglevel)


if __name__ == '__main__':
    main()
