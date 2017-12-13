from utils.utils import load_data, extract_trajectories, save_Q, initialize_save_data_folder, apply_phi_to_centroids
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration, iterate_policy
from irl.max_margin import run_max_margin
from irl.irl import  make_state_centroid_finder, make_phi, get_initial_state_distribution
from utils.plot import plot_margin_expected_value, plot_diff_feature_expectation, plot_value_function
from constants import *

import numpy as np
import pandas as pd
import time
import os
from multiprocessing import Pool


class ExperimentManager():
    def __init__(self, args):
        '''
        experiments: a list of Experiment instances
        '''
        # define hyperparameters
        self.num_trials = args.num_trials
        self.num_iterations = args.num_iterations
        self.svm_penalty = args.svm_penalty
        self.svm_epsilon = args.svm_epsilon
        self.use_pca = args.use_pca
        self.generate_new_data = args.generate_new_data
        self.num_bins = args.num_bins
        self.verbose = args.verbose
        self.experiment_name = time.strftime('%y%m%d_%H%M%S', time.gmtime())

        if self.verbose:
            print('num trials', self.num_trials)
            print('num iterations', self.num_iterations)
            print('svm penalty', self.svm_penalty)
            print('svm epsilon', self.svm_epsilon)
            print('')

        # loading data
        self.data = load_data(generate_new_data=self.generate_new_data)
        df_label = 'pca' if self.use_pca else 'original'
        self.df_train = self.data[df_label]['train']
        self.df_val = self.data[df_label]['val']
        self.df_centroids = self.data[df_label]['centroids']
        self.df = self.data[df_label]['full']

        if self.num_bins == 2:
            bins = ['min', '50%', 'max']
        else:
            bins = ['min', '25%', '50%', '75%', 'max']
        self.df_phi = apply_phi_to_centroids(self.df_centroids, self.df_train, bins=bins)
        self.phi = self.df_phi.as_matrix()
        assert np.all(np.isin(self.phi, [0, 1])), 'phi should be binary matrix'

        # build MDP
        # 1. bulld REWARD MATRIX
        self.features = self.df_phi.columns
        self.num_features = self.phi.shape[1]
        reward_matrix = np.zeros((NUM_STATES))
        reward_matrix[TERMINAL_STATE_ALIVE] = np.sqrt(self.num_features)
        reward_matrix[TERMINAL_STATE_DEAD]  = -np.sqrt(self.num_features)
        assert(np.isclose(np.sum(reward_matrix), 0))
        self.reward_matrix = reward_matrix


        # 2. build transition_matrix using full data
        t_path = TRAJECTORIES_PCA_FILEPATH if self.use_pca else TRAJECTORIES_FILEPATH
        tm_path = TRANSITION_MATRIX_PCA_FILEPATH if self.use_pca else TRANSITION_MATRIX_FILEPATH
        t_train_path = TRAIN_TRAJECTORIES_PCA_FILEPATH if self.use_pca else TRAIN_TRAJECTORIES_FILEPATH
        tm_train_path = TRAIN_TRANSITION_MATRIX_PCA_FILEPATH if self.use_pca else TRAIN_TRANSITION_MATRIX_FILEPATH

        trajectories = extract_trajectories(self.df, NUM_PURE_STATES, t_path)
        trajectory_train_ids = trajectories[:, 0]
        self.num_exp_trajectories = np.unique(trajectories[:, 0]).shape[0]
        transition_matrix, _ = \
                make_mdp(trajectories, NUM_STATES, NUM_ACTIONS, tm_path, REWARD_MATRIX_FILEPATH)
        print('number of expert trajectories for full data', self.num_exp_trajectories)
        assert np.isclose(np.sum(transition_matrix), NUM_STATES * NUM_ACTIONS), 'something wrong with \ test transition_matrix'
        self.transition_matrix = transition_matrix

        # 3. build transition_matrix using only training data
        trajectories_train = extract_trajectories(self.df_train, NUM_PURE_STATES, t_train_path)
        trajectory_train_ids = trajectories_train[:, 0]
        self.num_exp_trajectories_train = np.unique(trajectories_train[:, 0]).shape[0]
        transition_matrix_train, _ = \
                make_mdp(trajectories_train, NUM_STATES, NUM_ACTIONS, tm_train_path, TRAIN_REWARD_MATRIX_FILEPATH)
        print('number of expert trajectories for train data', self.num_exp_trajectories_train)
        assert np.isclose(np.sum(transition_matrix), NUM_STATES * NUM_ACTIONS), 'something wrong with \
             train transition_matrix'
        self.transition_matrix_train = transition_matrix_train

        # initial state sampler
        #self.sample_initial_state = make_initial_state_sampler(self.df_train)
        self.initial_state_probs = get_initial_state_distribution(self.df_train)

        # initalize saving folder
        self.save_path = initialize_save_data_folder()
        self.img_path = self.save_path + IMG_PATH
        print('will save expriment results to {}'.format(self.save_path))

        if self.verbose:
            print('number of features', self.num_features)
            print('transition_matrix_train size', self.transition_matrix_train.shape)
            print('reward_matrix size', self.reward_matrix.shape)
            print('max rewards: ', np.max(self.reward_matrix))
            print('min rewards: ', np.min(self.reward_matrix))
            print('max intermediate rewards: ', np.max(self.reward_matrix[:-2]))
            print('min intermediate rewards: ', np.min(self.reward_matrix[:-2]))
            print('')

        # init expert policies
        # derive expert policies from the full data
        # empirically there's little difference from when derived
        # only from training data
        Q_star = Q_value_iteration(transition_matrix, reward_matrix)
        self.pi_expert_phy_g = get_physician_policy(trajectories, is_stochastic=False)
        self.pi_expert_phy_s = get_physician_policy(trajectories, is_stochastic=True)
        save_Q(self.pi_expert_phy_g.Q, self.save_path, self.num_trials, self.num_iterations, PHYSICIAN_Q)

        self.pi_expert_mdp_g = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
        self.pi_expert_mdp_s = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
        save_Q(self.pi_expert_mdp_g.Q, self.save_path, self.num_trials, self.num_iterations, MDP_OPTIMAL_Q)

        # experiments
        self.experiments = []


    def save_experiment(self, res, exp_id, save_file_name):
        save_Q(res['approx_expert_Q'],
              self.save_path,
              self.num_trials,
              self.num_iterations,
              save_file_name)
        plot_margin_expected_value(res['margins'], self.num_trials, self.num_iterations, self.img_path, exp_id)
        plot_diff_feature_expectation(res['dist_mus'], self.num_trials, self.num_iterations, self.img_path, exp_id)
        plot_value_function(res['v_pis'], res['v_pi_expert'], self.num_trials, self.num_iterations, self.img_path, exp_id)


    def set_experiment(self, exp):
        exp.transition_matrix_train = self.transition_matrix_train
        exp.transition_matrix = self.transition_matrix
        exp.reward_matrix = self.reward_matrix
        exp.initial_state_probs = self.initial_state_probs
        exp.phi = self.phi
        exp.num_exp_trajectories = self.num_exp_trajectories
        exp.svm_penalty = self.svm_penalty
        exp.svm_epsilon = self.svm_epsilon
        exp.num_iterations = self.num_iterations
        exp.num_trials = self.num_trials
        exp.save_path = self.save_path
        exp.features = self.features
        exp.verbose = self.verbose

        self.experiments.append(exp)

    def _run(self, exp):
        res = exp.run()
        self.save_experiment(res, exp.experiment_id, exp.save_file_name)

    def run(self):
        '''
        parallelize the experiments
        '''
        pool = Pool(os.cpu_count() - 1)
        pool.map(self._run, self.experiments)

class Experiment():
    def __init__(self, experiment_id,
                 policy_expert,
                 save_file_name,
                 irl_use_stochastic_policy):
        self.experiment_id = experiment_id
        self.save_file_name = save_file_name
        self.pi_expert = policy_expert
        self.irl_use_stochastic_policy = irl_use_stochastic_policy

    def run(self):
        res = run_max_margin(self.transition_matrix_train,
                             self.transition_matrix,
                             self.reward_matrix,
                             self.pi_expert,
                             self.initial_state_probs,
                             self.phi,
                             self.num_exp_trajectories,
                             self.svm_penalty,
                             self.svm_epsilon,
                             self.num_iterations,
                             self.num_trials,
                             self.experiment_id,
                             self.save_path,
                             self.irl_use_stochastic_policy,
                             self.features,
                             self.verbose)

        return res


