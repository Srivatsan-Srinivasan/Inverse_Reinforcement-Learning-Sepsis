from mdp.solver import solve_mdp
# from policy.custom_policy import get_physician_policy
from policy.policy import GreedyPolicy, RandomPolicy
from irl.irl_gridworld import *
from optimize.quad_opt import QuadOpt
# from constants import *
from gridworld_project import *
from mdp.solver import Q_learning_solver_for_irl

import matplotlib.pyplot as plt
from scipy.optimize.minpack import curve_fit
import seaborn as sns
sns.set(style='dark', palette='husl')

# def max_margin_learner(df_cleansed, df_centroids, feature_columns,
#                    trajectories, transition_matrix, reward_matrix,
#                    num_iterations=50, epsilon=0.01):

def max_margin_learner(task, NUM_STATES, NUM_ACTIONS, transition_matrix, reward_matrix, pi_expert,
                   num_iterations=10, epsilon=0.01):
    '''
    reproduced maximum margin IRL algorithm
    described in Apprenticeship Learning paper (Abbeel and Ng, 2002)
    with Quadratic Programming
    '''
    # initialize utility functions and key variables
    # sample_initial_state = make_initial_state_sampler(df_cleansed)
    sample_initial_state = make_initial_state_sampler_mock(task)

    # get_state = make_state_centroid_finder(df_centroids, feature_columns) # get state centroid
    get_state = make_state_centroid_finder_mock(task, NUM_STATES)

    # phi = make_phi(task) # 
    opt = QuadOpt(epsilon)
    
    # step 0: initialize
    #np.random.seed(1)
    #alphas = np.ones(len(feature_columns))
    #W_expert = np.random.dirichlet(alphas, size=1)[0]
    #W_star = np.random.dirichlet(alphas, size=1)[0]

    # get pi_expert
    # pi_expert = get_physician_policy(trajectories)
    mu_pi_expert = estimate_feature_expectation(task, transition_matrix, sample_initial_state, get_state, phi, pi_expert)


    # get pi_star (mdp solution)
    # pi_star = solve_mdp(transition_matrix, np.mean(reward_matrix, axis=1))
    # mu_pi_star = estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, phi, pi_star)

    # step 1: initialize pi_tilda and mu_pi_tilda
    pi_tilda = RandomPolicy(NUM_PURE_STATES, NUM_ACTIONS)
    mu_pi_tilda = estimate_feature_expectation(task, transition_matrix,
                                                       sample_initial_state,
                                                       get_state, phi, pi_tilda)
    # initialize vars for plotting
    margins = []
    margins_star = []
    dist_mus = np.zeros(num_iterations)
    dist_mus_star = np.zeros(num_iterations)
    #v_pis = []
    #v_pis_star = []

    for i in range(num_iterations):
        print('iteration at {}/{} pi_expert'.format(i+1, num_iterations))
        # step 2: solve qp
        W_tilda, converged, margin = opt.optimize(mu_pi_expert, mu_pi_tilda)
        print ("Weight", W_tilda)

        # step 3: terminate if margin <= epsilon
        if not converged:
            # step 4: solve mdpr
            compute_reward = make_reward_computer(W_tilda, get_state, phi)
            reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
            pi_tilda, Q_table = Q_learning_solver_for_irl(task, transition_matrix, reward_matrix, NUM_STATES, NUM_ACTIONS)
            # pi_tilda = solve_mdp(transition_matrix, reward_matrix)
            # pi_tilda = pi_expert
            
            # step 5: estimate mu pi tilda
            mu_pi_tilda = estimate_feature_expectation(task, transition_matrix,
                                                       sample_initial_state,
                                                       get_state, phi, pi_tilda)

        
        # step 6: saving plotting vars
        dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
        dist_mus[i] = dist_mu
        margins.append(margin)

        print('dist_mus', dist_mus)
        print('margin', margins)
    import pdb;pdb.set_trace()
    OPT_POLICY = [np.argmax(i) for i in Q_table]
    print ("opt policy", OPT_POLICY)
        # print ("reward_matrix[9:18]", reward_matrix[9:18])
        # print ("reward_matrix[45:53]", reward_matrix[45:53])
        #v_pi = estimate_v_pi_tilda(W_expert, mu_pi_tilda, sample_initial_state)
        #v_pis.append(v_pi)

    # # the same process for mdp expert
    # for i in range(num_iterations):
    #     print('iteration at {}/{} for pi_star'.format(i+1, num_iterations))
    #     # step 2: solve qp
    #     W_star, converged, margin = opt.optimize(mu_pi_star, mu_pi_tilda)
        
    #     # step 3: terminate if margin <= epsilon
    #     if not converged:
    #         # step 4: solve mdpr
    #         compute_reward = make_reward_computer(W_star, get_state, phi)
    #         reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
    #         pi_tilda = solve_mdp(transition_matrix, reward_matrix)
    #         # step 5: estimate mu pi tilda
    #         mu_pi_tilda = estimate_feature_expectation(transition_matrix,
    #                                                    sample_initial_state,
    #                                                    get_state, phi, pi_tilda)

        # print('dist_mu', dist_mu)
        # print('margin', margin)
        # # step 6: saving plotting vars
        # dist_mu = np.linalg.norm(mu_pi_tilda - mu_pi_star, 2)
        # dist_mus_star[i] = dist_mu
        # margins_star.append(margin)

    fig = plt.figure(figsize=(10, 10))
    plt.plot(margins, label='vs. pi_expert')
    # plt.plot(margins_star, label='vs. pi_star')
    plt.xlabel('Number of iterations')
    plt.ylabel('SVM margin')
    plt.legend()
    plt.savefig('{}margin_{}_iter'.format(IMG_PATH, num_iterations), ppi=300, bbox_inches='tight')

    fig = plt.figure(figsize=(10, 10))
    plt.plot(dist_mus, label='vs. pi_expert')
    # plt.plot(dist_mus_star, label='vs. pi_star')
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

