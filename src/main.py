import numpy as np
from utils.utils import load_data, extract_trajectories, save_Q
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration, iterate_policy
from irl.max_margin import run_max_margin
from irl.irl import  make_state_centroid_finder, make_phi, make_initial_state_sampler
from constants import *


if __name__ == '__main__':
    # set hyperparams here
    num_trials = 10
    num_iterations = 2
    svm_penalty = 300.0
    svm_epsilon = 0.01
    verbose = True
    if verbose:
        print('num trials', num_trials)
        print('num iterations', num_iterations)
        print('svm penalty', svm_penalty)
        print('svm epsilon', svm_epsilon)
        print('')

    # loading the whole data
    df_train, df_val, df_centroids = load_data()

    feature_columns = df_centroids.columns
    trajectories = extract_trajectories(df_train, NUM_PURE_STATES, TRAIN_TRAJECTORIES_FILEPATH)
    trajectory_ids = trajectories[:, 0]
    num_exp_trajectories = np.unique(trajectories[:, 0]).shape[0]
    
    transition_matrix, reward_matrix = \
            make_mdp(trajectories, NUM_STATES, NUM_ACTIONS, TRAIN_TRANSITION_MATRIX_FILEPATH,
                     TRAIN_REWARD_MATRIX_FILEPATH)
    reward_matrix = np.zeros((NUM_STATES))
    # adjust rmax, rmin to keep w^Tphi(s) <= 1
    reward_matrix[TERMINAL_STATE_ALIVE] = np.sqrt(len(feature_columns))
    reward_matrix[TERMINAL_STATE_DEAD]  = -np.sqrt(len(feature_columns))

    # check irl/max_margin for implementation
    if verbose:
        print('number of features', len(feature_columns))
        print('transition_matrix size', transition_matrix.shape)
        print('reward_matrix size', reward_matrix.shape)
        print('max rewards: ', np.max(reward_matrix))
        print('min rewards: ', np.min(reward_matrix))
        print('max intermediate rewards: ', np.max(reward_matrix[:-2]))
        print('min intermediate rewards: ', np.min(reward_matrix[:-2]))
        print('')

    # initialize max margin irl stuff
    sample_initial_state = make_initial_state_sampler(df_train)
    get_state = make_state_centroid_finder(df_centroids, feature_columns)
    phi = make_phi(df_centroids)

    # extract empirical expert policy
    pi_expert_phy = get_physician_policy(trajectories, is_stochastic=False)
    pi_expert_phy = get_physician_policy(trajectories, is_stochastic=True)
    save_Q(pi_expert_phy.Q, PHYSICIAN_Q)

    experiment_id= 'maxmargin_empirical_expert'
    irl_physician_Q = run_max_margin(transition_matrix, np.copy(reward_matrix), pi_expert_phy,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, verbose)
    save_Q(irl_physician_Q, IRL_PHYSICIAN_Q)

    # extract artificial expert policy (optimal approximate MDP solution)
    Q_star = Q_value_iteration(transition_matrix, np.copy(reward_matrix))
    pi_expert_mdp = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
    save_Q(pi_expert_mdp.Q, MDP_OPTIMAL_Q)

    experiment_id= 'maxmargin_mdp_expert'
    irl_mdp_Q = run_max_margin(transition_matrix, np.copy(reward_matrix), pi_expert_mdp,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, verbose)
    save_Q(irl_mdp_Q, IRL_MDP_Q)

