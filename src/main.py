import numpy as np
#TODO: add logger
#import logging
#logger = logging.getLogger(__name__)

from utils.utils import load_data, extract_trajectories
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration
from irl.max_margin import run_max_margin
from irl.irl import  make_state_centroid_finder, make_phi, make_initial_state_sampler
from constants import NUM_STATES, NUM_ACTIONS, TERMINAL_STATE_ALIVE, TERMINAL_STATE_DEAD, NUM_PURE_STATES


if __name__ == '__main__':
    # set hyperparams here
    num_iterations = 2
    num_trials = 2
    svm_penalty = 300.0
    svm_epsilon = 0.01
    verbose = True

    # loading the whole data
    # TODO: load only training data
    # TODO: build training_mdp, test_mdp (IRL: model-based)
    df_train, df_val, df_centroids = load_data()

    feature_columns = df_centroids.columns
    trajectories = extract_trajectories(df_train, NUM_PURE_STATES)
    trajectory_ids = trajectories[:, 0]
    num_exp_trajectories = np.unique(trajectories[:, 0]).shape[0]
    
    transition_matrix, reward_matrix = make_mdp(trajectories, NUM_STATES, NUM_ACTIONS)
    
    # adjust rmax, rmin to keep w^Tphi(s) <= 1
    reward_matrix[TERMINAL_STATE_ALIVE] = np.sqrt(len(feature_columns))
    reward_matrix[TERMINAL_STATE_DEAD]  = -np.sqrt(len(feature_columns))
    ## make r(s, a) -> r(s)
    ## r(s) = E_pi_uniform[r(s,a)]
    reward_matrix = np.mean(reward_matrix, axis=1)

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
    pi_expert_g = get_physician_policy(trajectories, is_stochastic=False)
    pi_expert_s = get_physician_policy(trajectories, is_stochastic=True)
    experiment_id= 'maxmargin_empirical_expert'
    run_max_margin(transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, verbose)

    # extract artificial expert policy (optimal approximate MDP solution)
    Q_star = Q_value_iteration(transition_matrix, reward_matrix)
    pi_expert2 = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_star)
    experiment_id= 'maxmargin_artificial_expert'
    run_max_margin(transition_matrix, reward_matrix, pi_expert,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, verbose)


