# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:28:00 2017

@author: SrivatsanPC
"""

from utils import evaluation_utils as eu
from utils.utils import load_data, extract_trajectories
from constants import NUM_STATES, NUM_ACTIONS, TERMINAL_STATE_ALIVE, TERMINAL_STATE_DEAD, NUM_PURE_STATES
import numpy as np
import math

def test_code():
    print("Hello")
    
#opt_policy_learned should be a stochastic policy for each state, essentially a 750*25 array.
#phy_policy is single point estimate - one for each state - 750*1 array. phi_policy_stochastic is again 750*25 array.
def get_log_likelihood(df_cleansed,opt_policy_learned,phy_policy,phy_policy_stochastic,num_actions=25,no_int_id=0):
    trajs = extract_trajectories(df_cleansed, NUM_PURE_STATES)
    uni_trajs = np.unique(trajs[:, 0])
    LL = {}
    for traj_id in uni_trajs():
        path_taken = trajs[np.array([i[0]==traj_id for i in trajs])]
        #id,s,a,r,s'.Extract state,action
        path_taken_s_a = [path_taken[1:3]]
        target_acts = []
        target_acts_stoch = []
        candidate_acts = []
        for elem in path_taken_s_a:
            #elem[0]-state
            target_acts.append(phy_policy[elem[0]])
            target_acts_stoch.append(phy_policy_stochastic[elem[0]])
            candidate_acts.append(opt_policy_learned[elem[0]])
        LL[traj_id] = {"LL_IRL" : eu.log_likelihood(candidate_acts,target_acts),
                       "LL_random" : math.log((1/num_actions))* len(target_acts),
                       "LL_noint" : eu.log_likelihood_no_act(target_acts_stoch,no_int_id),
                       "length" : len(target_acts)}
    return LL

def get_KL_divergence(phy_policy_stochastic, opt_policy_learned, significant_states = range(750)):
    no_int_policy = eu.gen_single_action_probability()
    random_policy = eu.gen_random_probability()
    
    KL = {}
    for state in significant_states:
        phy_policy = phy_policy_stochastic[state]
        opt_policy = opt_policy_learned[state]
        KL[state] = {"IRL" : eu.KL_divergence(phy_policy,opt_policy),
                     "random" : eu.KL_divergence(phy_policy,random_policy),
                     "no_int_policy" : eu.KL_divergence(phy_policy,no_int_policy)                   
                    }
    return KL
        
        
        
        
    
    

            
            
        
        
        
    