import numpy as np
from scipy.stats import sem
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='dark', palette='husl')
  

from mdp.solver import Q_value_iteration
from policy.custom_policy import get_physician_policy
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from irl.irl import *
from optimize.quad_opt import QuadOpt
from constants import *

def max_margin_learner(transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, get_state, phi,
                       plot_prefix, num_iterations=20, num_trials=5):

    '''
    reproduced maximum margin IRL algorithm
    described in Apprenticeship Learning paper (Abbeel and Ng, 2002)
    with Quadratic Programming
    returns:
        margins = np.zeros((num_trials, num_iterations))
        dist_mus = np.zeros((num_trials, num_iterations))
        v_pis = np.zeros((num_trials, num_iterations))
    '''
    
    mu_pi_expert, v_pi_expert = estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, phi, pi_expert)
    print('objective: get close to ->')
    print('v_pi_expert', v_pi_expert)
    print('avg mu_pi_expert', np.mean(mu_pi_expert))

    # initialize vars for plotting
    margins = np.zeros((num_trials, num_iterations))
    dist_mus = np.zeros((num_trials, num_iterations))
    v_pis = np.zeros((num_trials, num_iterations))
    intermediate_reward_matrix = np.zeros((reward_matrix.shape))


    for trial_i in tqdm(range(num_trials)):
        print('max margin IRL starting ... with {}th trial'.format(1+trial_i))


        # step 1: initialize pi_tilda and mu_pi_tilda
        pi_tilda = RandomPolicy(NUM_STATES, NUM_ACTIONS)
        #old_pi_tilda = pi_tilda
        mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(transition_matrix,
                                                           sample_initial_state,
                                                           get_state, phi, pi_tilda)
        opt = QuadOpt(epsilon=0.001, penalty=300.0)
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
            compute_reward = make_reward_computer(W, get_state, phi)
            reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
            Q_star = Q_value_iteration(transition_matrix, reward_matrix)
            pi_tilda = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_star)
            pi_tildas.append(pi_tilda)
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
            print('')

            # step 5: estimate mu pi tilda
            mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(
                                   transition_matrix,
                                   sample_initial_state,
                                   get_state, phi, pi_tilda)
            print('avg mu_pi_tilda', np.mean(mu_pi_tilda))

            # step 6: saving plotting vars
            dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
            dist_mus[trial_i, i] = dist_mu
            margins[trial_i, i] = margin
            v_pis[trial_i, i] = v_pi_tilda
            print('dist_mu', dist_mu)
            print('margin', margin)
            print('v_pi', v_pi_tilda)
            intermediate_reward_matrix += reward_matrix
        # find a near-optimal policy from a policy reservoir
        # taken from Abbeel (2004)
        # TODO: retrieve near-optimal expert policy
        

    avg_margins = np.mean(margins, axis=0)
    avg_dist_mus = np.mean(dist_mus, axis=0)
    avg_v_pis = np.mean(v_pis, axis=0)
    avg_intermediate_reward_matrix = np.mean(intermediate_reward_matrix, axis=0)
    
    margin_se = sem(margins, axis=0)
    dist_se = sem(dist_mus, axis=0)
    v_pi_se = sem(v_pis, axis=0)
    

    fig = plt.figure(figsize=(10, 10))
    plt.ylim((0, np.max(margins) * 1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_margins,
                 label=r'$w^T\mu_E-w^T\mu_{\tilde{\pi}}$',
                 yerr=margin_se, fmt='-o')
    plt.xticks(np.arange(0, num_iterations+1, 5))
    plt.xlabel('Number of iterations')
    plt.ylabel('Margin')
    plt.legend()
    plt.savefig('{}{}_margin_i{}'.format(IMG_PATH, plot_prefix, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    plt.ylim((0, np.max(dist_mus) * 1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_dist_mus,
                 label=r'$||\mu_E-\mu_{\tilde{\pi}}||$',
                 yerr=dist_se, fmt='-o')
    plt.xticks(np.arange(1, num_iterations+1, 5))
    plt.xlabel('Number of iterations')
    plt.ylabel('Difference in feature expectation')
    plt.legend()
    plt.savefig('{}{}_dist_mu_i{}'.format(IMG_PATH, plot_prefix, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure(figsize=(10, 10))
    plt.ylim((v_pi_expert * 0.9, v_pi_expert * 1.1))
    plt.errorbar(np.arange(1, num_iterations+1), avg_v_pis, yerr=v_pi_se,
                 fmt='-o', label=r'$E_{s_0 \sim D(s)}[V^{\tilde \pi}(s_0)]$')
    plt.axhline(v_pi_expert, label=r'$E_{s_0 \sim D(s)}[V^{\pi_E}(s_0)]$', c='c')
    plt.xticks(np.arange(1, num_iterations+1, 5))
    plt.xlabel('Number of iterations')
    plt.ylabel('Performance')
    plt.legend()
    plt.savefig('{}{}_v_pi_i{}'.format(IMG_PATH, plot_prefix, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

    #fig = plt.figure(figsize=(10, 10))
    #im = plt.imshow(avg_intermediate_reward_matrix)
    #fig.colorbar(im)
