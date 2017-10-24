import numpy as np

from mdp.envs import GridWorld, mazeworld, simple_grid, testmaze
from mdp.dynamic_programming import iterate_policy, iterate_value, evaluate_policy
from policy.policy import EpsilonGreedyPolicy, GreedyPolicy, RandomPolicy

from policy.doubly_robust import eval_doubly_robust
from policy.weighted_doubly_robust import eval_weighted_doubly_robust
from policy.weighted_importance_sampling import eval_weighted_importance_sampling
from policy.importance_sampling import eval_importance_sampling
from policy.monte_carlo_on_policy import run_mc_actor
from policy.q_learning import Q_train

import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(palette="husl", style="dark")
palette = sns.color_palette("muted", 70)

MSE_RESULTS_MAZEWORLD = 'data_hw1/mse_results_mazeworld.npy'
FIG_OUTPUT_PATH_MAZEWORLD = 'data_hw1/mse_mazeworld'
PI_E_PATH_MAZEWORLD = 'data_hw1/pi_e_mazeworld.npy'
ALL_D_PATH_MAZEWORLD = 'data_hw1/all_d_mazeworld.npy'

MSE_RESULTS_CLIFFWORLD = 'data_hw1/mse_results_cliffworld.npy'
FIG_OUTPUT_PATH_CLIFFWORLD = 'data_hw1/mse_cliffworld'
PI_E_PATH_CLIFFWORLD = 'data_hw1/pi_e_cliffworld.npy'
ALL_D_PATH_CLIFFWORLD = 'data_hw1/all_d_cliffworld.npy'

''' TODO
[] plot on mazeworld, cliffworld
[] add the trial logic
[] plot the result with confidence interval
'''

def compute_mse(V_pi_e, V_hat_pi_e):
    # V_pi_e is fixed after policy evaluation
    # V_hat is num_episodes by 1 matrix or vector
    mse = np.power(V_hat_pi_e - V_pi_e, 2)
    return mse

def make_pi_evaluations(env, gamma):
    pi_evaluations = [] 

    # target policies to evaluate
    num_states = env.num_states
    num_actions = env.num_actions
    # target policy 1: random policy
    pi_random = RandomPolicy(num_states, num_actions)
    v_pi_random = evaluate_pi(env, pi_random, gamma)
    pi_evaluations.append(('random policy', pi_random, v_pi_random))

    # target policy 2: optimal policy (Q learning)
    eps = 0.1
    lr = 0.01
    num_episodes = 1000
    num_trial = 1
    pi_b = EpsilonGreedyPolicy(num_states, num_actions, eps)
    _, pi_q = Q_train(env, num_episodes, lr, gamma, pi_b)
    v_pi_q = evaluate_pi(env, pi_q, gamma)
    pi_evaluations.append(('optimal policy', pi_q, v_pi_q))

    # target policy 3: optimal policy (obatined solving MDP)
    #trainsition_matrix, reward_matrix = env.as_mdp()
    #Q = np.zeros((num_states, num_actions))
    #Q_star = iterate_policy(Q, trainsition_matrix, reward_matrix, max_iter=300)
    #pi_star = GreedyPolicy(num_states, num_actions, Q_star)
    #v_pi_star = evaluate_pi(env, pi_star, gamma)
    #pi_evaluations.append(('pi star from DP', pi_star, v_pi_star))

    return pi_evaluations


def generate_D(env, num_episodes, num_states, num_actions, eps, gamma, learning_rate=1e-2):
    pi_behavior = EpsilonGreedyPolicy(num_states, num_actions, eps)
    D, _ = Q_train(env, num_episodes, learning_rate, gamma, pi_behavior)
    return D


def evaluate_pi(env, pi_e, gamma):
    print('estimating the true V^pi_e truly on monte carlo rollouts')
    num_episodes = 100
    rewards = []
    for _ in range(num_episodes):
        exps = run_mc_actor(env, pi_e, max_local_iter=300)
        G = np.sum([(gamma**t)*e[2] for t, e in enumerate(exps)])
        rewards.append(G)
    return np.mean(rewards)

def setup_env(maze, reward_scheme, err_prob):
    env = GridWorld(maze=maze, terminal_markers=['*'], rewards=reward_scheme, action_error_prob=err_prob)
    num_actions = env.num_actions
    num_states = env.num_states
    return env, num_actions, num_states

def run_test(env, num_episodes, num_states, num_actions, env_name, all_d_path, pi_e_path,
        fig_output_path, mse_output_path):
    # hyperparameters
    num_episodes = num_episodes
    gamma = 0.99
    learning_rate = 1e-2
    # TODO: change this!
    epsilons = np.linspace(0, 1, 8)

    print('generating pi_evaluation')

    if os.path.isfile(pi_e_path):
        pi_evaluations = np.load(pi_e_path)
    else:
        pi_evaluations = make_pi_evaluations(env, gamma)
        pi_evaluations_np = np.asarray(pi_evaluations)
        np.save(pi_e_path, pi_evaluations_np)
    # solve mdp does not work...
    
    if os.path.isfile(all_d_path):
        all_D = np.load(all_d_path)
    else:
        all_D = []
        for eps in epsilons:
            # TODO: add trials
            print('generate D with eps = ', eps)
            D = generate_D(env, num_episodes, num_states, num_actions, eps, gamma)
            all_D.append(D)
        all_D = np.asarray(all_D)
        np.save(all_d_path, all_D)
    
    if os.path.isfile(mse_output_path):
        results = np.load(mse_output_path)
        print('mse result was found')
    else:
        results = []
        all_v_pi_e = []

        for i, D in enumerate(all_D):
            for pi_e_desc, pi_e, v_pi_e in pi_evaluations:
                print('evaluate {} over D'.format(pi_e_desc))
                print('computing V_hat_pi_e with WIS')
                WIS = eval_weighted_importance_sampling(env, D, pi_e, gamma)
                print('computing V_hat_pi_e with IS')
                IS = eval_importance_sampling(env, D, pi_e, gamma)
                print('computing v_hat_pi_e with DR')
                DR = eval_doubly_robust(env, D, pi_e, gamma)
                print('computing v_hat_pi_e with WDR')
                WDR = eval_weighted_doubly_robust(env, D, pi_e, gamma)
                ## compute MSE
                mse_wis = compute_mse(v_pi_e, WIS)
                mse_is = compute_mse(v_pi_e, IS)
                mse_dr = compute_mse(v_pi_e, DR)
                mse_wdr = compute_mse(v_pi_e, WDR)
                res = (mse_dr, mse_wdr, mse_wis, mse_is)
                results.append(res)
        np.save(mse_output_path, np.asarray(results))

    for i, _ in enumerate(all_D):
        for j, pi_e  in enumerate(pi_evaluations):
            pi_e_desc, _, _ = pi_e
            mse_dr, mse_wdr, mse_wis, mse_is = results[2*i+j]
            fig = plt.figure(figsize=(10, 10))
            fig.suptitle('Off-policy evaluation: {}'.format(pi_e_desc))
            plt.title('behavior pi: Q learning, epsilon: {}'.format(epsilons[i]))
            plt.plot(mse_dr, label='Doubly Robust')
            plt.plot(mse_wdr, label='Weighted Doubly Robust')
            plt.plot(mse_is, label='Importance Sampling')
            plt.plot(mse_wis, label='Weighted Importance Sampling')
            plt.xlabel('Number of episodes (D)')
            plt.ylabel('Mean squared error')
            plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
            plt.legend(loc=1, bbox_to_anchor=(1.3, 1))
            #plt.show()
            fig.savefig('{}_{}.png'.format(fig_output_path, 2*i + j), dpi=300, bbox_inches='tight')
            plt.close()


if __name__ == '__main__': 
    # set up an environment
    mazeworld_reward_scheme = {'*': 50, 'moved': -1, 'hit-wall': -1}
    err_prob = 0.2
    env, num_actions, num_states = setup_env(mazeworld, mazeworld_reward_scheme, err_prob)
    env_name = 'mazeworld'
    num_episodes = 50
    run_test(env, num_episodes, num_states, num_actions, env_name, ALL_D_PATH_MAZEWORLD, PI_E_PATH_MAZEWORLD, FIG_OUTPUT_PATH_MAZEWORLD, MSE_RESULTS_MAZEWORLD)

    # test on cliffworld
    cliffworld_reward_scheme = {'*': 50, 'moved': -1, 'hit-wall': -1, 'X': -50}
    err_prob = 0.2
    num_episodes = 50
    env, num_actions, num_states = setup_env(mazeworld, mazeworld_reward_scheme, err_prob)
    env_name = 'cliffworld'
    run_test(env, num_episodes, num_states, num_actions, env_name, ALL_D_PATH_CLIFFWORLD, PI_E_PATH_CLIFFWORLD, FIG_OUTPUT_PATH_CLIFFWORLD, MSE_RESULTS_CLIFFWORLD)


