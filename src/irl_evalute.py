import evaluation.log_likelihood as lh
from utils import evaluation_utils as eu
from utils.utils import load_data
import numpy as np
import pdb
from policy.policy import GreedyPolicy, StochasticPolicy
from constants import *


def test_against_expert(df, expert_filepath, irl_expert_filepath, plot_suffix=''):
    Q_star = np.load(expert_filepath)[:NUM_PURE_STATES+1, :]
    Q_irl = np.load(irl_expert_filepath)[:NUM_PURE_STATES+1, :]
    pi_physician_greedy = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star).get_opt_actions()
    pi_physician_stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star).query_Q_probs()
    opt_policy_learned = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_irl).query_Q_probs()

    LL = lh.get_log_likelihood(df, opt_policy_learned, pi_physician_greedy,
                               pi_physician_stochastic, num_states = NUM_PURE_STATES,
                               num_actions = NUM_ACTIONS, restrict_num=True, avg = True)
    KL = lh.get_KL_divergence(pi_physician_stochastic, opt_policy_learned)
    eu.plot_KL(KL, plot_suffix=plot_suffix)
    eu.plot_avg_LL(LL, plot_suffix=plot_suffix)


if __name__ == '__main__':
    df_train, df_val, df_centroids = load_data()
    # this will be appened to the plot filenames
    # all the plots will be saved to img/
    plot_phy_suffix = 'physician_test_1'
    plot_mdp_suffix = 'mdp_test_1'
    test_against_expert(df_train, PHYSICIAN_Q, IRL_PHYSICIAN_Q, plot_phy_suffix)
    test_against_expert(df_train, MDP_OPTIMAL_Q, IRL_MDP_Q, plot_mdp_suffix)
