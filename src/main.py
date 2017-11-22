from utils.utils import load_data, extract_trajectories

from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration
from irl.max_margin import max_margin_learner
from irl.irl import *
from constants import *


if __name__ == '__main__':
    num_iterations = 5
    num_trials = 10
    # loading the whole data
    df, df_cleansed, df_centroids = load_data()
    # decide which features to include
    feature_columns = df_centroids.columns
    trajectories = extract_trajectories(df_cleansed, NUM_PURE_STATES)
    transition_matrix, reward_matrix = make_mdp(trajectories, NUM_STATES, NUM_ACTIONS)
    
    # adjust rmax, rmin to keep w^Tphi(s) <= 1
    reward_matrix[TERMINAL_STATE_ALIVE] = np.sqrt(len(feature_columns))
    reward_matrix[TERMINAL_STATE_DEAD]  = -np.sqrt(len(feature_columns))
    ## make r(s, a) -> r(s)
    ## r(s) = E_pi_uniform[r(s,a)]
    reward_matrix = np.mean(reward_matrix, axis=1)

    # check irl/max_margin for implementation
    print('number of features', len(feature_columns))
    print('transition_matrix size', transition_matrix.shape)
    print('reward_matrix size', reward_matrix.shape)
    print('max rewards: ', np.max(reward_matrix))
    print('min rewards: ', np.min(reward_matrix))
    print('max intermediate rewards: ', np.max(reward_matrix[:-2]))
    print('min intermediate rewards: ', np.min(reward_matrix[:-2]))
    
    # initialize utility functions and key variables
    sample_initial_state = make_initial_state_sampler(df_cleansed)
    get_state = make_state_centroid_finder(df_centroids, feature_columns)
    phi = make_phi(df_centroids)
    #pi_expert = get_physician_policy(trajectories)
    #plot_prefix = 'exp'

    #max_margin_learner(transition_matrix, reward_matrix, pi_expert,
    #                   sample_initial_state, get_state, phi,
    #                   plot_prefix, num_iterations, num_trials)

    # get pi_star (mdp solution)
    print('getting pi_star and v_pi_star')
    Q_star = Q_value_iteration(transition_matrix, reward_matrix)
    pi_expert2 = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_star)
    plot_prefix2 = 'exp2'

    
    max_margin_learner(transition_matrix, reward_matrix, pi_expert2,
                       sample_initial_state, get_state, phi,
                       plot_prefix2, num_iterations, num_trials)


