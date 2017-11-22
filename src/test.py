import numpy as np
from policy.policy import GreedyPolicy
from mdp.solver import evaluate_policy_monte_carlo, Q_value_iteration, iterate_policy
from mdp.envs import mazeworld, GridWorld
from plot import *


if __name__ == '__main__':
    # # GRID WORLD
    mazeworld_reward_scheme = {'*': 1000, '.':-1, '#':-2}
    env = GridWorld(maze=mazeworld, terminal_markers=['*'], rewards=mazeworld_reward_scheme, action_error_prob=0.0)
    transition_matrix, reward_matrix = env.as_mdp(include_action_rewards=False)
    # slight modification to make solve_mdd work
    pi_star = Q_value_iteration(transition_matrix, reward_matrix, gamma=0.95)
    Q_star= iterate_policy(np.zeros((81, 4)), transition_matrix, reward_matrix, gamma=.95,)
    pi_star2 = GreedyPolicy(81, 4, Q_star)
    v_pi = evaluate_policy_monte_carlo(env, pi_star, gamma=0.95, num_episodes=100)
    v_pi2 = evaluate_policy_monte_carlo(env, pi_star2, gamma=0.95, num_episodes=100)



