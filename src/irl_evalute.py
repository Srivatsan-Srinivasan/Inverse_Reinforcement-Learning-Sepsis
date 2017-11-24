import os 
import numpy as np


import evaluation.log_likelihood as lh
from utils import evaluation_utils as eu
from utils.utils import load_data, initialize_save_data_folder
from policy.policy import GreedyPolicy, StochasticPolicy
from constants import *


def test_against_expert(df, expert_filepath, irl_expert_filepath, plot_suffix, img_path):
    Q_star = np.load(expert_filepath + '.npy')[:NUM_PURE_STATES+1, :]
    Q_irl = np.load(irl_expert_filepath + '.npy')[:NUM_PURE_STATES+1, :]
    pi_physician_greedy = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star).get_opt_actions()
    pi_physician_stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star).query_Q_probs()
    opt_policy_learned = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_irl).query_Q_probs()

    LL = lh.get_log_likelihood(df, opt_policy_learned, pi_physician_greedy,
                               pi_physician_stochastic, num_states = NUM_PURE_STATES,
                               num_actions = NUM_ACTIONS, restrict_num=True, avg = True)
    KL = lh.get_KL_divergence(pi_physician_stochastic, opt_policy_learned)
    
    print('will save plot results to {}'.format(img_path))
    eu.plot_KL(KL, plot_suffix=plot_suffix, save_path=img_path)
    eu.plot_avg_LL(LL, plot_suffix=plot_suffix, save_path=img_path)


if __name__ == '__main__':
    df_train, df_val, df_centroids = load_data()
    # for now, this must be set manually
    # example: date = '2016_02_02/'
    # make sure include trailing slash
    # check data folder to see for what dates data are available
    date = '2017_11_24/'
    if not os.path.exists(DATA_PATH + date):
        raise Exception('desired date should be specified for loading saved data.') 
    else:
        img_path = DATA_PATH + date + IMG_PATH

    # expert policies
    phy_q_filepath = DATA_PATH + date + PHYSICIAN_Q
    mdp_q_filepath = DATA_PATH + date + MDP_OPTIMAL_Q
    irl_phy_q_greedy_filepath = DATA_PATH + date + IRL_PHYSICIAN_Q_GREEDY
    irl_phy_q_stochastic_filepath = DATA_PATH + date + IRL_PHYSICIAN_Q_STOCHASTIC
    irl_mdp_q_greedy_filepath = DATA_PATH + date + IRL_MDP_Q_GREEDY
    irl_mdp_q_stochastic_filepath = DATA_PATH + date + IRL_MDP_Q_STOCHASTIC

    # this will be appened to the plot filenames
    # all the plots will be saved to img/
    plot_phy_greedy_id = 'physician_greedy'
    plot_mdp_greedy_id = 'mdp_greedy'
    plot_phy_stochastic_id = 'physician_stochastic'
    plot_mdp_stochastic_id = 'mdp_stochastic'
    test_against_expert(df_train, phy_q_filepath, irl_phy_q_greedy_filepath, plot_phy_greedy_id,
                        img_path)
    test_against_expert(df_train, phy_q_filepath, irl_phy_q_stochastic_filepath, plot_phy_greedy_id,
                       img_path)
    test_against_expert(df_train, mdp_q_filepath, irl_mdp_q_greedy_filepath, plot_mdp_greedy_id,
                        img_path)
    test_against_expert(df_train, mdp_q_filepath, irl_mdp_q_stochastic_filepath,
                        plot_mdp_stochastic_id, img_path)
