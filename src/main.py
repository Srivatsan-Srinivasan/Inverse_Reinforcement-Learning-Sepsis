import numpy as np


from utils.utils import load_data, extract_trajectories, save_Q, initialize_save_data_folder, apply_phi_to_centroids
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration, iterate_policy
from irl.max_margin import run_max_margin
from irl.irl import  make_state_centroid_finder, make_phi, make_initial_state_sampler
from utils.plot import plot_margin_expected_value, plot_diff_feature_expectation, plot_value_function
from constants import *

# don't move this for now
def plot_experiment(res, save_path, num_trials, num_iterations, img_path, experiment_id):
    plot_margin_expected_value(res['margins'], num_trials, num_iterations, img_path, experiment_id)
    plot_diff_feature_expectation(res['dist_mus'], num_trials, num_iterations, img_path, experiment_id)
    # TODO: this is not really a good measure of policy performance
    plot_value_function(res['v_pis'], res['v_pi_expert'], num_trials, num_iterations, img_path, experiment_id)
    # testing performance
    #plot_value_function(res['v_pis'], res['v_pi_expert'], num_trials, num_iterations, img_path, experiment_id)

if __name__ == '__main__':
    # set hyperparams here
    num_trials = 5
    num_iterations = 15
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

    # loading the data
    df_train, df_val, df_centroids, df = load_data()
    # initialize max margin irl stuff
    # preprocess phi
    sample_initial_state = make_initial_state_sampler(df_train)
    df_phi = apply_phi_to_centroids(df_centroids, df_train)
    phi = df_phi.as_matrix()
    assert np.all(np.isin(phi, [0, 1])), 'phi should be binary matrix'

    # build reward_matrix (not change whether train/val)
    features = df_phi.columns
    num_features = phi.shape[1]
    reward_matrix = np.zeros((NUM_STATES))
    # adjust rmax, rmin to keep w^Tphi(s) <= 1
    #TODO experiment
    reward_matrix[TERMINAL_STATE_ALIVE] = np.sqrt(num_features)
    reward_matrix[TERMINAL_STATE_DEAD]  = -np.sqrt(num_features)
    assert(np.isclose(np.sum(reward_matrix), 0))

    #evaluate_policy_monte_carlo()
    # build MDP using full data
    trajectories = extract_trajectories(df, NUM_PURE_STATES, TRAJECTORIES_FILEPATH)
    trajectory_train_ids = trajectories[:, 0]
    num_exp_trajectories = np.unique(trajectories[:, 0]).shape[0]
    transition_matrix, _ = \
            make_mdp(trajectories, NUM_STATES, NUM_ACTIONS, TRANSITION_MATRIX_FILEPATH,
                     REWARD_MATRIX_FILEPATH)
    print('number of expert trajectories for full data', num_exp_trajectories)
    assert np.isclose(np.sum(transition_matrix), NUM_STATES * NUM_ACTIONS), 'something wrong with \ test transition_matrix'

    # build MDP using only training data
    trajectories_train = extract_trajectories(df_train, NUM_PURE_STATES, TRAIN_TRAJECTORIES_FILEPATH)
    trajectory_train_ids = trajectories_train[:, 0]
    num_exp_trajectories_train = np.unique(trajectories_train[:, 0]).shape[0]
    transition_matrix_train, _ = \
            make_mdp(trajectories_train, NUM_STATES, NUM_ACTIONS, TRAIN_TRANSITION_MATRIX_FILEPATH,
                     TRAIN_REWARD_MATRIX_FILEPATH)
    print('number of expert trajectories for train data', num_exp_trajectories_train)
    assert np.isclose(np.sum(transition_matrix), NUM_STATES * NUM_ACTIONS), 'something wrong with \
         train transition_matrix'

    # show the world
    if verbose:
        print('number of features', num_features)
        print('transition_matrix_train size', transition_matrix_train.shape)
        print('reward_matrix size', reward_matrix.shape)
        print('max rewards: ', np.max(reward_matrix))
        print('min rewards: ', np.min(reward_matrix))
        print('max intermediate rewards: ', np.max(reward_matrix[:-2]))
        print('min intermediate rewards: ', np.min(reward_matrix[:-2]))
        print('')


    # initalize saving folder
    save_path = initialize_save_data_folder()
    img_path = save_path + IMG_PATH
    print('will save expriment results to {}'.format(save_path))
    #with open(save_path + 'experiment.txt', 'wb') as f:
    #    num_trials = 10
    #    num_iterations = 2
    #    svm_penalty = 300.0
    #    svm_epsilon = 0.01
    #    f.write()

    Q_star_train = Q_value_iteration(transition_matrix_train, reward_matrix)
    pi_expert_mdp_train = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star_train)
    Q_star = Q_value_iteration(transition_matrix, reward_matrix)
    pi_expert_mdp = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)

    # for now the experiments should be defined as below
    def experiment_1():
        print('EXPERIMENT #1: greedy deterministic clinician expert')
        experiment_id= 'greedy_physician'
        pi_expert_phy = get_physician_policy(trajectories, is_stochastic=False)
        save_Q(pi_expert_phy.Q, save_path, num_trials, num_iterations,  PHYSICIAN_Q)

        res = run_max_margin(transition_matrix_train, transition_matrix, reward_matrix, pi_expert_phy,
                           sample_initial_state, phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, experiment_id, save_path,
                            irl_use_stochastic_policy, features, verbose)

        save_Q(res['approx_expert_Q'], save_path, num_trials, num_iterations, IRL_PHYSICIAN_Q_GREEDY)
        plot_experiment(res, save_path, num_trials, num_iterations, img_path, experiment_id)


    def experiment_2():
        print('EXPERIMENT #2: clinician expert (stochastic)')
        experiment_id= 'stochastic_physician'
        pi_expert_phy_stochastic = get_physician_policy(trajectories, is_stochastic=True)
        res = run_max_margin(transition_matrix_train, transition_matrix, reward_matrix,
                                         pi_expert_phy_stochastic, sample_initial_state,
                                         phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                                         num_iterations, num_trials, experiment_id,
                                        save_path, irl_use_stochastic_policy, features, verbose)
        save_Q(res['approx_expert_Q'], save_path, num_trials, num_iterations, IRL_PHYSICIAN_Q_STOCHASTIC)
        plot_experiment(res, save_path, num_trials, num_iterations, img_path, experiment_id)


    def experiment_3():
        print('EXPERIMENT #3: artificial MDP-optimal expert')
        # used transition_matrix because expert can be as good as we want to be
        Q_star = Q_value_iteration(transition_matrix, reward_matrix)
        pi_expert_mdp = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
        save_Q(pi_expert_mdp.Q, save_path,num_trials, num_iterations, MDP_OPTIMAL_Q)

        experiment_id= 'greedy_mdp'
        res = run_max_margin(transition_matrix_train, transition_matrix, reward_matrix, pi_expert_mdp,
                           sample_initial_state, phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                           num_iterations, num_trials, experiment_id,
                          save_path, irl_use_stochastic_policy, features, verbose)
        save_Q(res['approx_expert_Q'], save_path, num_trials, num_iterations, IRL_MDP_Q_GREEDY)
        plot_experiment(res, save_path, num_trials, num_iterations, img_path, experiment_id)


    def experiment_4():
        print('EXPERIMENT #4: artificial MDP-optimal expert (stochastic)')
        # used transition_matrix because expert can be as good as we want to be
        Q_star = Q_value_iteration(transition_matrix, reward_matrix)
        pi_expert_mdp_stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)

        experiment_id= 'stochastic_mdp'
        res = run_max_margin(transition_matrix_train, transition_matrix, reward_matrix, pi_expert_mdp_stochastic,
                           sample_initial_state, phi, num_exp_trajectories, svm_penalty, svm_epsilon,
                           num_iterations, num_trials, experiment_id,
                           save_path, irl_use_stochastic_policy, features, verbose)
        save_Q(res['approx_expert_Q'], save_path, num_trials, num_iterations, IRL_MDP_Q_STOCHASTIC)
        plot_experiment(res, save_path, num_trials, num_iterations, img_path, experiment_id)

    # here run the experiments

    experiment_1()
    experiment_2()
    #experiment_3()
    #experiment_4()
