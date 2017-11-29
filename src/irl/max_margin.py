import numpy as np
from tqdm import tqdm

from mdp.solver import Q_value_iteration
from policy.custom_policy import get_physician_policy
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from irl.irl import *
from optimize.quad_opt import QuadOpt
from constants import NUM_STATES, NUM_ACTIONS, DATA_PATH

def _max_margin_learner(transition_matrix_train, transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, use_stochastic_policy, verbose):

    '''
    reproduced maximum margin IRL algorithm
    described in Apprenticeship Learning paper (Abbeel and Ng, 2002)
    with Quadratic Programming
    returns:
    results = {'margins': margins,
               'dist_mus': dist_mus,
               'v_pis': v_pis,
               'v_pi_expert': v_pi_expert,
               'svm_penlaty': svm_penalty,
               'svm_epsilon': svm_epsilon,
               'approx_expert_weights': approx_expert_weights,
               'num_exp_trajectories': num_exp_trajectories,
               'approx_expert_Q': approx_expert_Q
              }
    it is important we use only transition_matrix_train for training
    when testing, we will use transition_matrix, which is a better approximation of the world
    '''
    mu_pi_expert, v_pi_expert = estimate_feature_expectation(transition_matrix_train, sample_initial_state, phi, pi_expert)
    if verbose:
        print('objective: get close to ->')
        print('avg mu_pi_expert', np.mean(mu_pi_expert))
        print('v_pi_expert', v_pi_expert)
        print('')

    # initialize vars for plotting
    margins = np.full((num_trials, num_iterations), 10000)
    dist_mus = np.full((num_trials, num_iterations), 10000)
    v_pis = np.zeros((num_trials, num_iterations))
    intermediate_reward_matrix = np.zeros((reward_matrix.shape))
    approx_exp_policies = np.array([None] * num_trials)
    approx_exp_weights = np.array([None] * num_trials)


    for trial_i in tqdm(range(num_trials)):
        if verbose:
            print('max margin IRL starting ... with {}th trial'.format(1+trial_i))

        # step 1: initialize pi_tilda and mu_pi_tilda
        pi_tilda = RandomPolicy(NUM_PURE_STATES, NUM_ACTIONS)
        mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(transition_matrix_train,
                                                           sample_initial_state,
                                                           phi, pi_tilda)
        opt = QuadOpt(epsilon=svm_epsilon, penalty=svm_penalty)
        best_actions_old = None
        W_old = None
        pi_tildas = np.array([None]*num_iterations)
        weights = np.array([None]*num_iterations)
        for i in range(num_iterations):
            # step 2: solve qp
            W, converged, margin = opt.optimize(mu_pi_expert, mu_pi_tilda)
            # step 3: terminate if margin <= epsilon
            if converged:
                print('margin coverged with', margin)
                break

            weights[i] = W
            # step 4: solve mdpr
            compute_reward = make_reward_computer(W, phi)
            reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
            Q_star = Q_value_iteration(transition_matrix_train, reward_matrix)
            if use_stochastic_policy:
                pi_tilda = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
            else:
                pi_tilda = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
            pi_tildas[i] = pi_tilda
            # step 5: estimate mu pi tilda
            mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(
                                   transition_matrix_train,
                                   sample_initial_state,
                                   phi, pi_tilda)
            dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
            if verbose:
                # intermediate reeport for debugging
                print('max intermediate rewards: ', np.max(reward_matrix[:-2]))
                print('avg intermediate rewards: ', np.mean(reward_matrix[:-2]))
                print('min intermediate rewards: ', np.min(reward_matrix[:-2]))
                print('sd intermediate rewards: ', np.std(reward_matrix[:-2]))
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
                print('v_pi', v_pi_tilda)
                print('')

            # step 6: saving plotting vars
            dist_mus[trial_i, i] = dist_mu
            margins[trial_i, i] = margin
            v_pis[trial_i, i] = v_pi_tilda
            intermediate_reward_matrix += reward_matrix
        # find a near-optimal policy from a policy reservoir
        # taken from Abbeel (2004)
        # TODO: retrieve near-optimal expert policy
        min_margin_iter_idx = np.argmin(margins[trial_i])
        approx_exp_weights[trial_i] = weights[min_margin_iter_idx]
        approx_exp_policies[trial_i] = pi_tildas[min_margin_iter_idx].Q
        if verbose:
            print('best weights at {}th trial'.format(trial_i), weights[min_margin_iter_idx])
            print('best Q at {}th trial'.format(trial_i), pi_tildas[min_margin_iter_idx].Q)

    # there will be a better way to do a policy selection
    approx_expert_weights = np.mean(approx_exp_weights, axis=0)
    approx_expert_Q = np.mean(approx_exp_policies, axis=0)

    # test here
    #pi_irl_greedy = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, approx_expert_Q)
    #pi_irl_stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, approx_expert_Q)
    #v_expert = evaluate_policy_mc(transition_matrix, reward_matrix, sample_initial_state, pi_expert)
    #v_irl_greedy = evaluate_policy_mc(transition_matrix, reward_matrix, sample_initial_state, pi_irl_greedy)
    #v_irl_stochastic = evaluate_policy_mc(transition_matrix, reward_matrix, sample_initial_state, pi_irl_stochastic)
    #if verbose:
    #    print('')
    #    print('')
    #    print('')


    results = {'margins': margins,
               'dist_mus': dist_mus,
               'v_pis': v_pis,
               'v_pi_expert': v_pi_expert,
               'svm_penlaty': svm_penalty,
               'svm_epsilon': svm_epsilon,
               'approx_expert_weights': approx_expert_weights,
               'num_exp_trajectories': num_exp_trajectories,
               'approx_expert_Q': approx_expert_Q
              }
    return results


def run_max_margin(transition_matrix_train, transition_matrix, reward_matrix, pi_expert,
                    sample_initial_state,  phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, save_path, use_stochastic_policy, verbose):
    '''
    returns:
        approximate expert policy
    '''

    res = _max_margin_learner(transition_matrix_train, transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, use_stochastic_policy, verbose)

    np.save('{}{}_t{}xi{}_result'.format(save_path, experiment_id, num_trials, num_iterations), res)
    np.save('{}{}_t{}xi{}_weights'.format(save_path, experiment_id, num_trials, num_iterations),
            res['approx_expert_weights'])
    print('final weights learned for ', experiment_id)
    print(res['approx_expert_weights'])
    print('')
    return res

