from mdp.solver import solve_mdp, evaluate_policy, iterate_policy
from policy.custom_policy import get_physician_policy
from policy.policy import GreedyPolicy, RandomPolicy
from irl.irl import *
from optimize.quad_opt import QuadOpt
from constants import *
import numpy as np


from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize.minpack import curve_fit
import seaborn as sns
sns.set(style='dark', palette='husl')

def max_margin_learner(df_cleansed, df_centroids, feature_columns,
                   trajectories, transition_matrix, reward_matrix,
                   num_iterations=50, num_trials=20, epsilon=0.01):
    
    '''
    reproduced maximum margin IRL algorithm
    described in Apprenticeship Learning paper (Abbeel and Ng, 2002)
    with Quadratic Programming
    '''
    # initialize utility functions and key variables
    sample_initial_state = make_initial_state_sampler(df_cleansed)
    get_state = make_state_centroid_finder(df_centroids, feature_columns)
    phi = make_phi(df_centroids)
    opt = QuadOpt(epsilon)
    
    # step 0: initialize
    #np.random.seed(1)
    #alphas = np.ones(len(feature_columns))
    #W_expert = np.random.dirichlet(alphas, size=1)[0]
    #W_star = np.random.dirichlet(alphas, size=1)[0]

    # get pi_expert
    print('getting pi_expert and v_pi_expert')
    pi_expert = get_physician_policy(trajectories)
    mu_pi_expert, v_pi_expert = estimate_feature_expectation(transition_matrix,
                                                sample_initial_state, get_state, phi, pi_expert)
    #print('v_pi_expert', v_pi_expert)

    # get pi_star (mdp solution)
    print('getting pi_star and v_pi_star')
    #pi_star = solve_mdp(transition_matrix, np.mean(reward_matrix, axis=1))
    #mu_pi_star, v_pi_tilda = estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, phi, pi_star)

    
    # initialize vars for plotting
    margins = np.zeros((num_trials, num_iterations))
    margins_star = np.zeros((num_trials, num_iterations))
    dist_mus = np.zeros((num_trials, num_iterations))
    dist_mus_star = np.zeros((num_trials, num_iterations))
    v_pis = np.zeros((num_trials, num_iterations))
    v_pis_star = np.zeros((num_trials, num_iterations))
    
    for trial_i in tqdm(range(num_trials)):
        print('learning R(s) for pi_expert')
        # step 1: initialize pi_tilda and mu_pi_tilda
        pi_tilda = RandomPolicy(NUM_STATES, NUM_ACTIONS)
        mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(transition_matrix,
                                                           sample_initial_state,
                                                           get_state, phi, pi_tilda)
        for i in range(num_iterations):
            # step 2: solve qp
            W_expert, converged, margin = opt.optimize(mu_pi_expert, mu_pi_tilda)
            
            # step 3: terminate if margin <= epsilon
            if not converged:
                # step 4: solve mdpr
                compute_reward = make_reward_computer(W_expert, get_state, phi)
                reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
                Q_tilda = iterate_policy(np.zeros((NUM_STATES, NUM_ACTIONS)), transition_matrix, reward_matrix)
                pi_tilda = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_tilda)
                #pi_tilda = solve_mdp(transition_matrix, reward_matrix)
                
                # step 5: estimate mu pi tilda
                mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(transition_matrix,
                                       sample_initial_state,
                                       get_state, phi, pi_tilda)

            # step 6: saving plotting vars
            dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
            dist_mus[trial_i, i] = dist_mu
            margins[trial_i, i] = margin
            v_pis[trial_i, i] = v_pi_tilda
            print('dist_mu', dist_mu)
            print('margin', margin)
            print('v_pi', v_pi_tilda)

    avg_margins = np.mean(margins, axis=0)
    max_margins = np.amax(margins, axis=0)
    min_margins = np.amin(margins, axis=0)
    
    avg_dist_mus = np.mean(dist_mus, axis=0)
    max_dist_mus = np.amax(dist_mus, axis=0)
    min_dist_mus = np.amin(dist_mus, axis=0)

    avg_v_pis = np.mean(v_pis, axis=0)
    max_v_pis = np.amax(v_pis, axis=0)
    min_v_pis = np.amin(v_pis, axis=0)
    
    
    for trial_i in tqdm(range(num_trials)):
        print('learning R(s) of pi_star')
        # step 1: initialize pi_tilda and mu_pi_tilda
        pi_tilda = RandomPolicy(NUM_STATES, NUM_ACTIONS)
        mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(transition_matrix,
                                                           sample_initial_state,
                                                           get_state, phi, pi_tilda)
        # the same process for mdp expert
        for i in range(num_iterations):
            print('iteration at {}/{} for pi_star'.format(i+1, num_iterations))
            # step 2: solve qp
            W_star, converged, margin = opt.optimize(mu_pi_star, mu_pi_tilda)
            
            # step 3: terminate if margin <= epsilon
            if not converged:
                # step 4: solve mdpr
                compute_reward = make_reward_computer(W_star, get_state, phi)
                reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
                pi_tilda = solve_mdp(transition_matrix, reward_matrix)
                # step 5: estimate mu pi tilda
                mu_pi_tilda, v_pi_tilda = estimate_feature_expectation(transition_matrix,
                                                           sample_initial_state,
                                                           get_state, phi, pi_tilda)

            # step 6: saving plotting vars
            dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_star, 2)
            dist_mus_star[trial_i, i] = dist_mu
            margins_star[trial_i, i] = margin
            v_pis_star[trial_i, i] = v_pi_tilda
            print('dist_mu', dist_mu)
            print('margin', margin)
            print('v_pi', v_pi_tilda)


    avg_margins_star = np.mean(margins_star, axis=0)
    max_margins_star = np.amax(margins_star, axis=0)
    min_margins_star = np.amin(margins_star, axis=0)
    
    avg_dist_mus_star = np.mean(dist_mus_star, axis=0)
    max_dist_mus_star = np.amax(dist_mus_star, axis=0)
    min_dist_mus_star = np.amin(dist_mus_star, axis=0)

    avg_v_pis_star = np.mean(v_pis_star, axis=0)
    max_v_pis_star = np.amax(v_pis_star, axis=0)
    min_v_pis_star = np.amin(v_pis_star, axis=0)

    import pdb;pdb.set_trace()

    fig = plt.figure(figsize=(10, 10))
    plt.plot(margins, label='vs. pi_expert')
    plt.plot(margins_star, label='vs. pi_star')
    plt.xlabel('Number of iterations')
    plt.ylabel('SVM margin')
    plt.legend()
    plt.savefig('{}margin_{}_iter'.format(IMG_PATH, num_iterations), ppi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(10, 10))
    plt.plot(dist_mus, label='vs. pi_expert')
    plt.plot(dist_mus_star, label='vs. pi_star')
    plt.xlabel('Number of iterations')
    plt.ylabel("Distance to the expert's feature expectation")
    plt.legend()
    plt.savefig('{}distance_{}_iter'.format(IMG_PATH, num_iterations), ppi=300, bbox_inches='tight')

    #fig = plt.figure(figsize=(10, 10))
    #plt.plot(v_pis, label='vs. performance_expert')
    #plt.plot(v_pis_star, label='vs. perfomrance_star')
    #plt.xlabel('Number of iterations')
    #plt.ylabel('v_pi_tilda compared to v_pi_expert/star')
    #plt.legend()
    #plt.savefig('{}value_{}_iter'.format(IMG_PATH, num_iterations), ppi=300, bbox_inches='tight')

