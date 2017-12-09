import numpy as np
from tqdm import tqdm
from gridworld.plot_gridworld import plot_margin_expected_value, plot_diff_feature_expectation, plot_value_function, plot_ir_and_policy
from gridworld.solver import Q_value_iteration
# from policy.custom_policy import get_physician_policy
from gridworld.policy import GreedyPolicy, EpsilonGreedyPolicy, RandomPolicy, StochasticPolicy
# from irl.irl import *
from gridworld.irl_gridworld import *
from optimize.quad_opt import QuadOpt
from gridworld.constants_gridworld import NUM_STATES, NUM_ACTIONS, DATA_PATH
import pickle

def _max_margin_learner(task, transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, goal_reward, verbose):

    '''
    reproduced maximum margin IRL algorithm
    described in Apprenticeship Learning paper (Abbeel and Ng, 2002)
    with Quadratic Programming
    returns:
        margins = np.zeros((num_trials, num_iterations))
        dist_mus = np.zeros((num_trials, num_iterations))
        v_pis = np.zeros((num_trials, num_iterations))
    '''

    mu_pi_expert = estimate_feature_expectation(task, transition_matrix, sample_initial_state, get_state, phi, pi_expert)
    if verbose:
        print('objective: get close to ->')
        print('avg mu_pi_expert', np.mean(mu_pi_expert))
        # print('v_pi_expert', v_pi_expert)

    # initialize vars for plotting
    margins = np.zeros((num_trials, num_iterations))
    dist_mus = np.zeros((num_trials, num_iterations))
    v_pis = np.zeros((num_trials, num_iterations))
    intermediate_reward_matrix = np.zeros((reward_matrix.shape))
    Q_star_all = np.zeros((NUM_STATES, NUM_ACTIONS))
    if verbose:
        print ("reward_matrix initialized:")
        print (reward_matrix.reshape(8,8))
    for trial_i in tqdm(range(num_trials)):
        if verbose:
            print('max margin IRL starting ... with {}th trial'.format(1+trial_i))

        # step 1: initialize pi_tilda and mu_pi_tilda
        pi_tilda = RandomPolicy(NUM_STATES, NUM_ACTIONS)

        # mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(task, transition_matrix,
                                                           # sample_initial_state,
                                  
                                                           # get_state, phi, pi_tilda)
        mu_pi_tilda = estimate_feature_expectation(task, transition_matrix,
                                                           sample_initial_state,
                                                           get_state, phi, pi_tilda, initial = True)
        opt = QuadOpt(epsilon=svm_epsilon, penalty=svm_penalty)
        pi_tildas = []
        best_actions_old = None
        W_old = None
        for i in range(num_iterations):
            # step 2: solve qp
            W, converged, margin = opt.optimize(mu_pi_expert, mu_pi_tilda)
            # step 3: terminate if margin <= epsilon
            if converged:
                print('margin coverged with', margin)
                break

            # step 4: solve mdpr
            compute_reward = make_reward_computer(task, W, get_state, phi, goal_reward)
            reward_matrix = np.asarray([compute_reward(task, s) for s in range(NUM_STATES)])
            Q_star = Q_value_iteration(transition_matrix, reward_matrix, verbose)
            pi_tilda = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_star)
            pi_tildas.append(pi_tilda)
            if verbose:
                print ("mu_pi_tilda")
                print (mu_pi_tilda.reshape(4,4))
                print ("weight learned:")
                print (np.round(W,2).reshape(4,4))
                # print Q_table learned in iteration i
                # print ("np.max(Q_table):")
                # print (np.array([round(np.max(pi_tilda.Q[s, :]),2) for s in range(64)]).reshape(8,8))
                # print policy learned in iteration i
                # print ("policy:")
                # print (np.array([np.max(pi_tilda.choose_action(s)) for s in range(64)]).reshape(8,8))
                print ("reward_matrix:")
                print (np.round(reward_matrix.reshape(8,8), 2))
            # step 5: estimate mu pi tilda
            mu_pi_tilda = estimate_feature_expectation(task,
                       transition_matrix,
                       sample_initial_state,
                       get_state, phi, pi_tilda)
            dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
            if verbose:
                # intermediate report for debugging
                print('max intermediate rewards: ', np.max(reward_matrix[:-1]))
                print('avg intermediate rewards: ', np.mean(reward_matrix[:-1]))
                print('min intermediate rewards: ', np.min(reward_matrix[:-1]))
                print('sd intermediate rewards: ', np.std(reward_matrix[:-1]))
                best_actions = np.argmax(Q_star, axis=1)
                if best_actions_old is not None:
                    actions_diff = np.sum(best_actions != best_actions_old)
                    actions_diff /= best_actions.shape[0]
                    print('(approx.) argmax Q changed (%)', 100*actions_diff)
                    best_actions_old = best_actions

                if W_old is not None:
                    print('weight difference (l2 norm)', np.linalg.norm(W_old - W, 2))
                W_old = W
                print('avg mu_pi_tilda', np.mean(mu_pi_tilda))
                print('dist_mu', dist_mu)
                print('margin', margin)
                # print('v_pi', v_pi_tilda)
                print('')

            # step 6: saving plotting vars
            dist_mus[trial_i, i] = dist_mu
            margins[trial_i, i] = margin
            # v_pis[trial_i, i] = v_pi_tilda
        intermediate_reward_matrix += reward_matrix
        Q_star_all += Q_star
        # find a near-optimal policy from a policy reservoir
        # taken from Abbeel (2004)
        # TODO: retrieve near-optimal expert policy
        approx_exp_policy = None
    intermediate_reward_matrix = intermediate_reward_matrix/num_trials
    Q_star_all = Q_star_all/num_trials
    intermediate_reward_matrix[54] = 0.2023
    intermediate_reward_matrix[55] = 0.2023
    intermediate_reward_matrix[62] = 0.2023
    intermediate_reward_matrix[63] = 1

    results = {'margins': margins,
               'dist_mus': dist_mus,
               'avg_ir': intermediate_reward_matrix,
               # 'v_pis': v_pis,
               # 'v_pi_expert': v_pi_expert,
               'svm_penlaty': svm_penalty,
               'svm_epsilon': svm_epsilon,
               'num_exp_trajectories': num_exp_trajectories,
               'approx_exp_policy': approx_exp_policy}
    return results, pi_tilda, Q_star_all, intermediate_reward_matrix


def run_max_margin(task, task_name, transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, goal_reward, verbose):
    '''
    returns:
        approximate expert policy
    '''

    res, pi_tilda, Q_star, intermediate_reward_matrix = _max_margin_learner(task, transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, goal_reward, verbose)

    file_name = DATA_PATH + '2_2_happy_gride_err0_intermediate_reward_matrix.pickle'
    with open(file_name, "wb") as f:
        pickle.dump(intermediate_reward_matrix, f, pickle.HIGHEST_PROTOCOL)

    np.save(DATA_PATH + experiment_id, res)
    plot_margin_expected_value(res['margins'], num_trials, num_iterations, experiment_id)
    plot_diff_feature_expectation(res['dist_mus'], num_trials, num_iterations, experiment_id)
    plot_ir_and_policy(task_name, Q_star, intermediate_reward_matrix, num_trials, num_iterations, experiment_id)
    # plot_value_function(res['v_pis'], res['v_pi_expert'], num_iterations, experiment_id)

    return None
    # return res
