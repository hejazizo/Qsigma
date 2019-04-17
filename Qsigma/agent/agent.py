import logging
import math

import numpy as np

import copy

logger = logging.getLogger(__name__)


class QsigmaAgent():
    def __init__(self, config: str):
        self.config = config

    def run(self, env, policy, num_episodes):

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
            'map': None,
            'PI': None,
            'Q': None
        }

        # -------------------- #
        # ----- "S" ARSA ----- #
        # -------------------- #
        initial_state = env.position['START']
        episode['states'].append(initial_state)

        # --------------------- #
        # ----- S "A" RSA ----- #
        # --------------------- #
        initial_action = policy.choose_action(env.position['START'], env.actions, PI, Q)
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
                next_state, reward, episode_finished = env.act(episode['states'][t], episode['actions'][t])
                episode['states'].append(next_state)
                episode['rewards'] += reward
                episode['steps'] += 1

                if episode_finished:
                    T = t+1
                    episode['targets'].append(reward-episode['q'][t])
                else:
                    # -------------------- #
                    # ----- SARS "A" ----- #
                    # -------------------- #
                    next_action = policy.choose_action(episode['states'][t+1], env.actions, PI, Q)
                    episode['actions'].append(next_action)

                    # select sigma according to method and Qsigma mode
                    sigma = self.choose_sigma(method, q_mode=q_mode, last_sigma=episode['sigmas'][-1], base=base, t=t)
                    episode['sigmas'].append(sigma)
                    episode['q'].append(Q[next_state, next_action])
                    episode['pi'].append(PI[next_state, next_action])

                    target = reward + sigma*gamma*episode['q'][t+1] + (1-sigma)*gamma*np.dot(PI[next_state, :], Q[next_state, :]) - episode['q'][t]
                    episode['targets'].append(target)

            # STATE OF TERMINATION
            tau = t - n_steps + 1
            if tau >= 0:
                E = 1
                G = episode['q'][tau]
                for k in range(tau, min(tau+n_steps-1, T-1)+1):
                    G += E*episode['targets'][k]
                    E *= gamma*((1-episode['sigmas'][k])*episode['pi'][k] + episode['sigmas'][k])

                # Update Q function
                Q[episode['states'][tau], episode['actions'][tau]] += alpha*(G - Q[episode['states'][tau], episode['actions'][tau]])
                # Update policy to be epsilon-greedy w.r.t. Q
                policy.make_greedy(episode['states'][tau], env.actions, PI, Q)

            if tau == T-1:
                break

        episode['PI'] = PI.copy()
        episode['Q'] = Q.copy()

        # episode['map'] = env.world
        logger.info(f'\n {env.world}')
        logger.info(f"Episode finished with: {episode['steps']} steps")
        return episode

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
            "ExpectedSARSA": 0,
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
                sigma[method]["alt"] = 1 - last_sigma
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

            elif q_mode == 'episode_state':
                sigma[method]["dec"] = 0

            return sigma[method][q_mode]

        return sigma[method]
