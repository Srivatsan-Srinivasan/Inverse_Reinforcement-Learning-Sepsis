from utils.utils import load_data, extract_trajectories, save_Q, initialize_save_data_folder, apply_phi_to_centroids
from utils.evaluation_utils import plot_KL, plot_avg_LL
from policy.policy import GreedyPolicy, RandomPolicy, StochasticPolicy
from policy.custom_policy import get_physician_policy
from mdp.builder import make_mdp
from mdp.solver import Q_value_iteration, iterate_policy
from irl.max_margin import run_max_margin
from irl.irl import  make_state_centroid_finder, make_phi, get_initial_state_distribution
from utils.plot import *
from constants import *
import evaluation.log_likelihood as lh

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
        self.parallelized = args.parallelized
        self.hyperplane_margin = args.hyperplane_margin
        self.num_exp_trajectories = args.num_exp_trajectories

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

        # some utility stats
        self.avg_mortality_per_state = pd.Series(np.zeros(NUM_PURE_STATES), name='avg_mortality')
        g_mortality = self.df.groupby(['state'])['died_in_hosp'].value_counts().groupby(level=0)
        am = g_mortality.apply(lambda x : 100* x / float(x.sum()) )[:, 1]
        self.avg_mortality_per_state[am.index] = am
        self.avg_mortality_per_state = self.avg_mortality_per_state.tolist()

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
        transition_matrix, _ = \
                make_mdp(trajectories, NUM_STATES, NUM_ACTIONS, tm_path, REWARD_MATRIX_FILEPATH)
        assert np.isclose(np.sum(transition_matrix), NUM_STATES * NUM_ACTIONS), 'something wrong with \ test transition_matrix'
        self.transition_matrix = transition_matrix

        # 3. build transition_matrix using only training data
        trajectories_train = extract_trajectories(self.df_train, NUM_PURE_STATES, t_train_path)
        transition_matrix_train, _ = \
                make_mdp(trajectories_train, NUM_STATES, NUM_ACTIONS, tm_train_path, TRAIN_REWARD_MATRIX_FILEPATH)
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
        self.pi_expert_phy_g = get_physician_policy(trajectories, is_stochastic=False)
        self.pi_expert_phy_s = get_physician_policy(trajectories, is_stochastic=True)
        save_Q(self.pi_expert_phy_g.Q, self.save_path, self.num_trials, self.num_iterations, PHYSICIAN_Q)

        Q_star = Q_value_iteration(self.transition_matrix, self.reward_matrix)
        self.pi_expert_mdp_g = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
        self.pi_expert_mdp_s = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
        save_Q(self.pi_expert_mdp_g.Q, self.save_path, self.num_trials, self.num_iterations, MDP_OPTIMAL_Q)


        # experiments
        self.experiments = []
        self.pi_phy_vaso = None
        self.pi_phy_vaso_sd_l = None
        self.pi_phy_vaso_sd_r = None
        self.pi_phy_iv = None
        self.pi_phy_iv_sd_l = None
        self.pi_phy_iv_sd_r = None
        self.pi_mdp_vaso_probs = None
        self.pi_mdp_iv_probs = None
        self.pi_irl_vaso_probs = None
        self.pi_irl_iv_probs = None


        # self:
        df = self.df
        iv_min = df['action_iv'].min()
        iv_max = df['action_iv'].max()
        vaso_min = df['action_vaso'].min()
        vaso_max = df['action_vaso'].max()
        df.loc[df['action_iv_sd_left'] != df['action_iv'], 'action_iv_sd_left'] = (df['action_iv'].copy() - 1).clip(iv_min, iv_max)
        df.loc[df['action_iv_sd_right'] != df['action_iv'], 'action_iv_sd_right'] = (df['action_iv'].copy() + 1).clip(iv_min, iv_max)
        df.loc[df['action_vaso_sd_left'] != df['action_vaso'], 'action_vaso_sd_left'] = (df['action_vaso'].copy() - 1).clip(vaso_min, vaso_max)
        df.loc[df['action_vaso_sd_right'] != df['action_vaso'], 'action_vaso_sd_right'] = (df['action_vaso'].copy() + 1).clip(vaso_min, vaso_max)
        self.df = df


    def save_experiment(self, res, exp):
        save_Q(res['approx_expert_Q'],
              exp.save_path,
              exp.num_trials,
              exp.num_iterations,
              exp.save_file_name)

        feature_importances = res['feature_imp']
        top10_pos = feature_importances[:10]
        top10_neg = feature_importances[-10:]
        if self.verbose:
            print('final weights learned for ', exp.experiment_id)
            print('top 10 positive weights', top10_pos)
            print('top 10 negative weights', top10_neg)
            print('')
        np.save('{}{}_t{}xi{}_result'.format(exp.save_path,
                                             exp.experiment_id,
                                             exp.num_trials,
                                             exp.num_iterations), res)
        np.save('{}{}_t{}xi{}_weights'.format(exp.save_path,
                                              exp.experiment_id,
                                              exp.num_trials,
                                              exp.num_iterations),
                                              res['approx_expert_weights'])

        plot_margin_expected_value(res['margins'],
                                   self.num_trials,
                                   self.num_iterations,
                                   self.img_path,
                                   exp.experiment_id)

        # @refactor this is not needed
        #plot_diff_feature_expectation(res['dist_mus'],
        #                              self.num_trials,
        #                              self.num_iterations,
        #                              self.img_path,
        #                              exp.experiment_id)

        plot_value_function(res['v_pis'],
                            res['v_pi_expert'],
                            self.num_trials,
                            self.num_iterations,
                            self.img_path,
                            exp.experiment_id)

        plot_intermediate_rewards_vs_mortality(res['intermediate_rewards'],
                                               self.avg_mortality_per_state,
                                               self.img_path,
                                               exp.experiment_id,
                                               exp.num_trials,
                                               exp.num_iterations)

        if exp.irl_use_stochastic_policy:
            df = self.df_train[self.df_centroids.columns]
            pi_irl_s = StochasticPolicy(NUM_PURE_STATES,
                                        NUM_ACTIONS,
                                        res['approx_expert_Q'])

            if self.pi_phy_vaso is None or \
                self.pi_phy_iv is None or \
                self.pi_mdp_vaso_probs is None or \
                self.pi_mdp_iv_probs is None or \
                self.pi_irl_vaso_probs is None or \
                self.pi_irl_iv_probs is None:
                # do this only one time
                self._prepare_deviation_plots(self.df,
                                              self.pi_expert_mdp_s,
                                              pi_irl_s)


            sd_vaso = self.df[['action_vaso_sd_left', 'action_vaso', 'action_vaso_sd_right']].drop_duplicates().as_matrix()
            plot_deviation_from_experts(sd_vaso,
                                        self.pi_phy_vaso,
                                        self.pi_mdp_vaso_probs,
                                        self.pi_irl_vaso_probs,
                                        self.img_path,
                                        exp.experiment_id + '_vaso',
                                        exp.num_trials,
                                        exp.num_iterations)

            sd_iv = self.df[['action_iv_sd_left', 'action_iv', 'action_iv_sd_right']].drop_duplicates().as_matrix()
            plot_deviation_from_experts(sd_iv,
                                        self.pi_phy_iv,
                                        self.pi_mdp_iv_probs,
                                        self.pi_irl_iv_probs,
                                        self.img_path,
                                        exp.experiment_id + '_iv',
                                        exp.num_trials,
                                        exp.num_iterations)
            # kl and loglikelihood
            LL = lh.get_log_likelihood(df,
                                       pi_irl_s,
                                       self.pi_expert_phy_g,
                                       self.pi_expert_phy_s,
                                       num_states = NUM_PURE_STATES,
                                       num_actions = NUM_ACTIONS,
                                       restrict_num = True,
                                       avg = True)

            KL = lh.get_KL_divergence(self.pi_expert_phy_s,
                                      pi_irl_s)
            plot_KL(KL,
                   plot_suffix=exp.experiment_id,
                   save_path=self.img_path,
                   show=False,
                   iter_num=exp.num_iterations,
                   trial_num=exp.num_trials)

            plot_avg_LL(LL,
                       plot_suffix=exp.experiment_id,
                       save_path=self.img_path,
                       show=False,
                       iter_num=exp.num_iterations,
                       trial_num=exp.num_trials)

    def _prepare_deviation_plots(self, df, pi_mdp, pi_irl):
        path_vaso = TRAJECTORIES_PCA_VASO_FILEPATH if self.use_pca else TRAJECTORIES_VASO_FILEPATH
        #path_vaso_sd_l = TRAJECTORIES_PCA_VASO_SD_LEFT_FILEPATH if self.use_pca else TRAJECTORIES_VASO_SD_LEFT_FILEPATH
        #path_vaso_sd_r = TRAJECTORIES_PCA_VASO_SD_RIGHT_FILEPATH if self.use_pca else TRAJECTORIES_VASO_SD_RIGHT_FILEPATH
        path_iv = TRAJECTORIES_PCA_IV_FILEPATH if self.use_pca else TRAJECTORIES_IV_FILEPATH
        #path_iv_sd_l = TRAJECTORIES_PCA_IV_SD_LEFT_FILEPATH if self.use_pca else TRAJECTORIES_IV_SD_LEFT_FILEPATH
        #path_iv_sd_r = TRAJECTORIES_PCA_IV_SD_RIGHT_FILEPATH if self.use_pca else TRAJECTORIES_IV_SD_RIGHT_FILEPATH

        traj_vaso = extract_trajectories(self.df, NUM_PURE_STATES, path_vaso, 'action_vaso')
        #traj_vaso_sd_l = extract_trajectories(self.df, NUM_PURE_STATES, path_vaso_sd_l, 'action_vaso_sd_left')
        #traj_vaso_sd_r = extract_trajectories(self.df, NUM_PURE_STATES, path_vaso_sd_r, 'action_vaso_sd_right')
        traj_iv = extract_trajectories(self.df, NUM_PURE_STATES, path_iv, 'action_iv')
        #traj_iv_sd_l = extract_trajectories(self.df, NUM_PURE_STATES, path_iv_sd_l, 'action_iv_sd_left')
        #traj_iv_sd_r = extract_trajectories(self.df, NUM_PURE_STATES, path_iv_sd_r, 'action_iv_sd_right')
        num_bins = 5
        self.pi_phy_vaso = get_physician_policy(traj_vaso, num_actions=num_bins, is_stochastic=True)
        #self.pi_phy_vaso_sd_l = get_physician_policy(traj_vaso_sd_l, num_actions=num_bins, is_stochastic=True)
        #self.pi_phy_vaso_sd_r = get_physician_policy(traj_vaso_sd_r, num_actions=num_bins, is_stochastic=True)
        self.pi_phy_iv = get_physician_policy(traj_iv, num_actions=num_bins, is_stochastic=True)
        #self.pi_phy_iv_sd_l = get_physician_policy(traj_iv_sd_l, num_actions=num_bins, is_stochastic=True)
        #self.pi_phy_iv_sd_r = get_physician_policy(traj_iv_sd_r, num_actions=num_bins, is_stochastic=True)

        num_bins = 5
        mdp_probs = pi_mdp.query_Q_probs()
        mdp_vaso_probs = np.zeros((NUM_PURE_STATES, num_bins))
        mdp_iv_probs = np.zeros((NUM_PURE_STATES, num_bins))
        irl_probs = pi_irl.query_Q_probs()
        irl_vaso_probs = np.zeros((NUM_PURE_STATES, num_bins))
        irl_iv_probs = np.zeros((NUM_PURE_STATES, num_bins))
        vaso_bins = np.arange(25) // num_bins
        iv_bins = np.arange(25) % num_bins

        for bin_i in range(num_bins):
            # marginalize out iv and vaso
            mdp_vaso_probs[:, bin_i] = np.sum(mdp_probs[:, np.flatnonzero(vaso_bins == bin_i)], axis=1)
            mdp_iv_probs[:, bin_i] = np.sum(mdp_probs[:, np.flatnonzero(iv_bins == bin_i)], axis=1)
            irl_vaso_probs[:, bin_i] = np.sum(irl_probs[:, np.flatnonzero(vaso_bins == bin_i)], axis=1)
            irl_iv_probs[:, bin_i] = np.sum(irl_probs[:, np.flatnonzero(iv_bins == bin_i)], axis=1)

        assert np.isclose(np.sum(mdp_vaso_probs), NUM_PURE_STATES)
        assert np.isclose(np.sum(mdp_iv_probs), NUM_PURE_STATES)
        assert np.isclose(np.sum(irl_vaso_probs), NUM_PURE_STATES)
        assert np.isclose(np.sum(irl_iv_probs), NUM_PURE_STATES)
        self.pi_mdp_vaso_probs = mdp_vaso_probs
        self.pi_mdp_iv_probs = mdp_iv_probs
        self.pi_irl_vaso_probs = irl_vaso_probs
        self.pi_irl_iv_probs = irl_iv_probs


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
        exp.hyperplane_margin = self.hyperplane_margin
        if self.use_pca:
            exp.experiment_id += '_pca'
            exp.save_file_name += '_pca'
        self.experiments.append(exp)

    def _run(self, exp):
        res = exp.run()
        self.save_experiment(res, exp)

    def run(self):
        '''
        parallelize the experiments
        '''
        if self.parallelized:
            try:
                pool = Pool(os.cpu_count() - 1)
                pool.map(self._run, self.experiments)
            finally:
                pool.close()
                pool.join()
        else:
            for e in self.experiments:
                self._run(e)


    def _run_perf_vs_trajectories(self, num_exp_trajectories):

        pi_irl_s = StochasticPolicy(NUM_PURE_STATES,
                                    NUM_ACTIONS,
                                    res['approx_expert_Q'])
        #res = exp.run()

        #v_pi_irl_g, v_pi_irl_s, num_exp_trajectories
        pass

    def run_perf_vs_trajectories(self, max_num_exp_trajectories=3000):
        step_size = 50
        num_exp_trajectories_list = np.arange(1, max_num_exp_trajectories+1, step_size)
        v_pi_irl_gs = np.zeros(num_exp_trajectories_list.shape)
        v_pi_irl_ss = np.zeros(num_exp_trajectories_list.shape)
        if self.parallelized:
            try:
                pool = Pool(os.cpu_count() - 1)
                v_pi_irl_g, v_pi_irl_s, num_exp_trajectories = \
                        pool.map(self._run_perf_vs_trajectories, num_exp_trajectories_list)
                v_pi_irl_gs[num_exp_trajectories // step_size] = v_pi_irl_g
                v_pi_irl_ss[num_exp_trajectories // step_size] = v_pi_irl_s
            finally:
                pool.close()
                pool.join()
        else:
            raise Exception('turn on parallelization')
        res = {'v_pi_irl_gs': v_pi_irl_gs, 'v_pi_irl_ss': v_pi_irl_ss}
        cur_t = time.strftime('%y%m%d_%H%M%S', time.gmtime())
        np.save('{}perf_vs_traj_result_{}'.format(self.save_path, cur_t), res)

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
                             self.irl_use_stochastic_policy,
                             self.features,
                             self.hyperplane_margin,
                             self.verbose)

        return res


