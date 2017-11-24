import evaluation.log_likelihood as lh
from utils import evaluation_utils as eu
from utils.utils import load_data
import numpy as np
import pdb
from policy.policy import GreedyPolicy, StochasticPolicy
from constants import *


#Q_star = np.array([[5 for i in range(NUM_ACTIONS)]for j in range(NUM_PURE_STATES)])
#Q_star = np.load(PHYSICIAN_Q)
Q_star = np.load(MDP_OPTIMAL_Q)
greedy = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
pi_physician_greedy = greedy.get_opt_actions()
pi_physician_stochastic = stochastic.query_Q_probs()[:NUM_PURE_STATES, :]
pdb.set_trace()
if not np.isclose(np.sum(pi_physician_stochastic), NUM_PURE_STATES):
    pdb.set_trace()

#opt_policy_learned = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, np.load(IRL_PHYSICIAN_Q)).query_Q_probs()
opt_policy_learned = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, np.load(IRL_MDP_Q)).query_Q_probs()

df_train, df_val, df_centroids = load_data()
LL = lh.get_log_likelihood(df_train, opt_policy_learned, pi_physician_greedy,
                           pi_physician_stochastic, num_states = NUM_PURE_STATES,
                           num_actions = NUM_ACTIONS,restrict_num=True, avg = True)
KL = lh.get_KL_divergence(pi_physician_stochastic,opt_policy_learned)
eu.plot_KL(KL)
eu.plot_avg_LL(LL)
assert(True)
