# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:32:19 2017

@author: SrivatsanPC
"""

import evaluation.log_likelihood as lh
from utils import evaluation_utils as eu
from utils.utils import load_data
import numpy as np
import pdb
from policy.policy import GreedyPolicy, StochasticPolicy
from constants import NUM_ACTIONS,  NUM_PURE_STATES
Q_star = np.array([[5 for i in range(NUM_ACTIONS)]for j in range(NUM_PURE_STATES)])
greedy = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
stochastic = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_star)
pi_physician_greedy = greedy.get_opt_actions()
pi_physician_stochastic = stochastic.get_stochastic_actions()

opt_policy_learned = np.ones((NUM_PURE_STATES,NUM_ACTIONS)) * 0.04
df, df_cleansed, df_centroids = load_data()
LL = lh.get_log_likelihood(df_cleansed, opt_policy_learned, pi_physician_greedy, 
                           pi_physician_stochastic, num_states = NUM_PURE_STATES, 
                           num_actions = NUM_ACTIONS,restrict_num=True, avg = True)
KL = lh.get_KL_divergence(pi_physician_stochastic,opt_policy_learned)
eu.plot_KL(KL)
eu.plot_avg_LL(LL)
pdb.set_trace()
assert(True)