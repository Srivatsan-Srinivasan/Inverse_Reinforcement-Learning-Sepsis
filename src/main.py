import numpy as np
from utils.utils import load_data, extract_trajectories, save_Q, initialize_save_data_folder
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration, iterate_policy
from irl.max_margin import run_max_margin
from irl.irl import  make_state_centroid_finder, make_phi, make_initial_state_sampler
from constants import *


if __name__ == '__main__':
    # set hyperparams here
    num_trials = 30
    num_iterations = 30
    svm_penalty = 300.0
    svm_epsilon = 1e-4
    irl_use_stochastic_policy = False
    #irl_use_stochastic_policy = True
    # e.g. custom_experiment_id = '_some_name_starting_with_underscore'
    # if you use this, you will need to manually specify the filepath when loading them
    custom_experiment_id = ''
    #custom_experiment_id = '_greedy_irl'
    verbose = False
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

    # initalize saving
    save_path = initialize_save_data_folder()
    print('will save expriment results to {}'.format(save_path))
    #with open(save_path + 'experiment.txt', 'wb') as f:
    #    num_trials = 10
    #    num_iterations = 2
    #    svm_penalty = 300.0
    #    svm_epsilon = 0.01
    #    f.write()

    print('EXPERIMENT #1: greedy deterministic clinician expert')
    experiment_id= 'greedy_physician' + custom_experiment_id
    pi_expert_phy = get_physician_policy(trajectories, is_stochastic=False)
    save_Q(pi_expert_phy.Q, save_path, PHYSICIAN_Q)

    irl_physician_Q_greedy = run_max_margin(transition_matrix, np.copy(reward_matrix), pi_expert_phy,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                   num_iterations, num_trials, experiment_id, save_path,
                        irl_use_stochastic_policy, verbose)
    save_Q(irl_physician_Q_greedy, save_path, IRL_PHYSICIAN_Q_GREEDY)

    print('EXPERIMENT #2: stochastic clinician expert')
    experiment_id= 'stochastic_physician' + custom_experiment_id
    pi_expert_phy_stochastic = get_physician_policy(trajectories, is_stochastic=True)
    irl_physician_Q_stochastic = run_max_margin(transition_matrix, np.copy(reward_matrix),
                                     pi_expert_phy_stochastic, sample_initial_state,
                                     get_state, phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                                     num_iterations, num_trials, experiment_id,
                                    save_path, irl_use_stochastic_policy, verbose)
    save_Q(irl_physician_Q_stochastic, save_path, IRL_PHYSICIAN_Q_STOCHASTIC)

    print('EXPERIMENT #3: artificial MDP-optimal expert')
    Q_star = Q_value_iteration(transition_matrix, np.copy(reward_matrix))
    pi_expert_mdp = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
    save_Q(pi_expert_mdp.Q, save_path, MDP_OPTIMAL_Q)

    experiment_id= 'greedy_mdp' + custom_experiment_id
    irl_mdp_Q_greedy = run_max_margin(transition_matrix, np.copy(reward_matrix), pi_expert_mdp,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, experiment_id,
                      save_path, irl_use_stochastic_policy, verbose)
    save_Q(irl_mdp_Q_greedy, save_path, IRL_MDP_Q_GREEDY)

    print('EXPERIMENT #4: artificial MDP-optimal expert (stochastic)')
    Q_star = Q_value_iteration(transition_matrix, np.copy(reward_matrix))
    pi_expert_mdp_stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)

    experiment_id= 'stochastic_mdp' + custom_experiment_id
    irl_mdp_Q_stochastic = run_max_margin(transition_matrix, np.copy(reward_matrix), pi_expert_mdp_stochastic,
                       sample_initial_state, get_state, phi,
                       num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, experiment_id,
                       save_path, irl_use_stochastic_policy, verbose)
    save_Q(irl_mdp_Q_stochastic, save_path, IRL_MDP_Q_STOCHASTIC)

