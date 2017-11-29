import os
import numpy as np
import pandas as pd
import pprint as pp

import evaluation.log_likelihood as lh
from utils import evaluation_utils as eu
from utils.utils import load_data, initialize_save_data_folder
from policy.policy import GreedyPolicy, StochasticPolicy
from constants import *


def test_against_expert(df, expert_filepath, irl_expert_filepath,
                        plot_suffix, img_path, date, trial_num, iter_num, verbose=False):
    q_star_path = '{}_t{}xi{}.npy'.format(expert_filepath, trial_num, iter_num)
    irl_path = '{}_t{}xi{}.npy'.format(irl_expert_filepath, trial_num, iter_num)
    irl_weights_path = '{}_t{}xi{}_weights.npy'.format(DATA_PATH + date + plot_suffix, trial_num, iter_num)

    Q_star = np.load(q_star_path)[:NUM_PURE_STATES+1, :]
    Q_irl = np.load(irl_path)[:NUM_PURE_STATES+1, :]
    Q_irl_weights = np.load(irl_weights_path)

    if verbose:
        pp.pprint('loading saved data trained w/ iterations of {}'.format(iter_num))
        pp.pprint("{}'s Q:\n{}".format(plot_suffix, Q_star))
        pp.pprint("IRL expert's Q\n{}".format(Q_irl))
        pp.pprint("IRL expert Q's weights")
        weights = pd.Series(data=Q_irl_weights, index=df.columns)
        pp.pprint(weights)


    pi_physician_greedy = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star).get_opt_actions()
    pi_physician_stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star).query_Q_probs()
    opt_policy_learned = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_irl).query_Q_probs()

    LL = lh.get_log_likelihood(df, opt_policy_learned, pi_physician_greedy,
                               pi_physician_stochastic, num_states = NUM_PURE_STATES,
                               num_actions = NUM_ACTIONS, restrict_num=True, avg = True)
    KL = lh.get_KL_divergence(pi_physician_stochastic, opt_policy_learned)

    print('will save plot results to {}'.format(img_path))
    eu.plot_KL(KL, plot_suffix=plot_suffix, save_path=img_path, show=False, iter_num=iter_num,
               trial_num=trial_num)
    eu.plot_avg_LL(LL, plot_suffix=plot_suffix, save_path=img_path, show=False, iter_num=iter_num,
                   trial_num=trial_num)


if __name__ == '__main__':
    df_train, df_val, df_centroids, df_full = load_data()
    df = df_train[df_centroids.columns]
    # for now, this must be set manually
    # example: date = '2016_02_02/'
    # make sure include trailing slash
    # check data folder to see for what dates data are available
    date = '2017_11_29/'
    # pick which models you want manually for now...
    trial_num = 5
    iter_num = 20
    if not os.path.exists(DATA_PATH + date):
        raise Exception('desired date should be specified for loading saved data.')
    else:
        img_path = DATA_PATH + date + IMG_PATH

    # expert policies
    greedy_phy_q_filepath = DATA_PATH + date + PHYSICIAN_Q
    stochastic_phy_q_filepath = DATA_PATH + date + PHYSICIAN_Q
    greedy_mdp_q_filepath = DATA_PATH + date + MDP_OPTIMAL_Q
    stochastic_mdp_q_filepath = DATA_PATH + date + MDP_OPTIMAL_Q
    irl_phy_q_greedy_filepath = DATA_PATH + date + IRL_PHYSICIAN_Q_GREEDY
    irl_phy_q_stochastic_filepath = DATA_PATH + date + IRL_PHYSICIAN_Q_STOCHASTIC
    irl_mdp_q_greedy_filepath = DATA_PATH + date + IRL_MDP_Q_GREEDY
    irl_mdp_q_stochastic_filepath = DATA_PATH + date + IRL_MDP_Q_STOCHASTIC

    # this will be appened to the plot filenames
    # all the plots will be saved to img/
    plot_phy_greedy_id = 'greedy_physician'
    plot_phy_stochastic_id = 'stochastic_physician'
    plot_mdp_greedy_id = 'greedy_mdp'
    plot_mdp_stochastic_id = 'stochastic_mdp'
    test_against_expert(df, greedy_phy_q_filepath, irl_phy_q_greedy_filepath,
                        plot_phy_greedy_id, img_path, date, trial_num, iter_num)
    test_against_expert(df, stochastic_phy_q_filepath, irl_phy_q_stochastic_filepath,
                        plot_phy_stochastic_id, img_path, date, trial_num, iter_num)
    test_against_expert(df, greedy_mdp_q_filepath, irl_mdp_q_greedy_filepath,
                        plot_mdp_greedy_id, img_path, date, trial_num, iter_num)
    test_against_expert(df, stochastic_mdp_q_filepath, irl_mdp_q_stochastic_filepath,
                        plot_mdp_stochastic_id, img_path, date, trial_num, iter_num)
