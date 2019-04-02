import logging

from rasa_nlu import utils
from rasa_nlu.utils import read_yaml_file
import numpy as np
from Qsigma.environment.grid_world import GridWorld
from Qsigma.policy.eps_greedy import EpsGreedy

logger = logging.getLogger(__name__)


class QsigmaAgent():
    def __init__(self, config: str):
        self.experiments = []
        self.config = config

    def run(self, env, policy, num_experiments, num_episodes):
        for num_exp in range(num_experiments):
            logger.info(f'Experiment #{num_exp+1} started...')
            experiment = self._run_experiment(env, policy, num_episodes)
            self.experiments.append(experiment)

    def _run_experiment(self, env, policy, num_episodes):

        PI = np.zeros((len(env.states), len(env.actions)))  # policy: <state, action> -> <float>
        Q = np.zeros((len(env.states), len(env.actions)))   # <state, action> -> <float>

        experiment = []
        for num_eps in range(num_episodes):
            logger.info(f'Episode #{num_eps+1} started...')
            episode = self._run_episode(PI, Q, env, policy, **self.config)
            experiment.append(episode)

        return experiment

    def _run_episode(self, PI, Q, env, policy, method, q_mode=None, base=None, gamma=0.99, n_steps=5, alpha=0.1):
        env.reset()

        episode = {
            'steps': 0,         # number of steps to finish episode
            'rewards': 0,       # total reward for episode
            'states': [],       # visited states
            'actions': [],      # actions taken
            'q': [],            # q value: q[t] = Q[state[t], action[t]]
            'pi': [],           # actions probability
            'sigmas': [0],      # selected sigma
            'targets': [],      # target reward
            'map': None
        }

        # -------------------- #
        # ----- "S" ARSA ----- #
        # -------------------- #
        initial_state = env.position['START']
        episode['states'].append(initial_state)

        # --------------------- #
        # ----- S "A" RSA ----- #
        # --------------------- #
        initial_action = policy.choose_action(env.position['START'], PI, Q)
        episode['actions'].append(initial_action)

        episode['q'].append(Q[initial_state, initial_action])
        episode['pi'].append(PI[initial_action, initial_action])

        # RUNNING an episode
        T = np.inf
        t = -1

        while True:
            t += 1
            assert len(episode['actions']) == len(episode['q']) == len(episode['pi']) == len(episode['sigmas'])

            if t < T:
                # ------------------------ #
                # ----- SA "R" "S" A ----- #
                # ------------------------ #
                next_state, reward = env.move(episode['states'][t], episode['actions'][t])
                episode['states'].append(next_state)
                episode['rewards'] += reward
                episode['steps'] += 1

                if next_state == env.position['GOAL']:
                    T = t+1
                    episode['targets'].append(reward-episode['q'][t])
                else:
                    # -------------------- #
                    # ----- SARS "A" ----- #
                    # -------------------- #
                    next_action = policy.choose_action(episode['states'][t+1], PI, Q)
                    episode['actions'].append(next_action)

                    # select sigma according to method and Qsigma mode
                    sigma = env.choose_sigma(method, q_mode=q_mode, last_sigma=episode['sigmas'][-1], base=base, t=t)
                    episode['sigmas'].append(sigma)
                    episode['q'].append(Q[next_state, next_action])

                    target = reward + sigma*gamma*episode['q'][t+1] + (1-sigma)*gamma*np.sum(PI[next_state, :]*Q[next_state, :]) - episode['q'][t]
                    episode['targets'].append(target)
                    episode['pi'].append(PI[next_state, next_action])

            # STATE OF TERMINATION
            tau = t - n_steps + 1
            if tau >= 0:
                E = 1
                G = episode['q'][tau]
                for k in range(tau, min(tau+n_steps-1, T-1)):
                    G += E*episode['targets'][k]
                    E *= gamma*((1-episode['sigmas'][k+1])*episode['pi'][k+1] + episode['sigmas'][k+1])

                # Update Q function
                Q[episode['states'][tau], episode['actions'][tau]] += alpha*(G - Q[episode['states'][tau], episode['actions'][tau]])
                # Update policy to be epsilon-greedy w.r.t. Q
                policy.make_greedy(episode['states'][tau], PI, Q)

            if tau == T-1:
                break

        episode['map'] = env.map
        logger.warning(env.map)
        logger.info(f"Episode finished with: {episode['steps']} steps")
        return episode


def create_argument_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Q-sigma Reinforcement Learning Algorithm')

    parser.add_argument('-c', '--config', type=str, required=True, help='Q-sigma config file.')
    parser.add_argument('--n_episode', type=int, required=True, help='Number of episodes.')
    parser.add_argument('--n_experiment', type=int, required=True, help='Number of episodes.')

    utils.add_logging_option_arguments(parser)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    utils.configure_colored_logging(args.loglevel)
    config = read_yaml_file(args.config)

    env = GridWorld(config=config['environment'])
    policy = EpsGreedy(config=config['policy'])
    qsigma = QsigmaAgent(config=config['agent'])
    qsigma.run(env, policy, num_experiments=args.n_experiment, num_episodes=args.n_episode)

    logger.info(f"Steps: {[[episode['steps'] for episode in experiment] for experiment in qsigma.experiments]}")
    logger.info(f"Rewards: {[[episode['rewards'] for episode in experiment] for experiment in qsigma.experiments]}")


if __name__ == '__main__':
    main()
