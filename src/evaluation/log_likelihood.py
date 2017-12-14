from utils import evaluation_utils as eu
from utils.utils import load_data, extract_trajectories
from constants import *
import numpy as np
import math
import pdb

#opt_policy_learned should be a stochastic policy for each state, essentially a 750*25 array.
#phy_policy is single point estimate - one for each state - 750*1 array. phi_policy_stochastic is again 750*25 array.
def get_log_likelihood(df, opt_policy_learned, phy_policy, phy_policy_stochastic, num_states = NUM_PURE_STATES, num_actions=25, no_int_id=0, restrict_num = False, avg = False):
    opt_policy_learned = opt_policy_learned.query_Q_probs()
    phy_policy = phy_policy.get_opt_actions()
    phy_policy_stochastic = phy_policy_stochastic.query_Q_probs()
    trajs = extract_trajectories(df, num_states, TRAIN_TRAJECTORIES_FILEPATH)
    uni_trajs = np.unique(trajs[:, 0])
    LL = {}
    if restrict_num:
        uni_trajs = uni_trajs[:100]
    for traj_id in uni_trajs:
        path_taken = trajs[np.array([i[0]==traj_id for i in trajs])]
        #id,s,a,r,s'.Extract state,action
        path_taken_s_a = [p[1:3] for p in path_taken]
        target_acts = []
        target_acts_stoch = []
        candidate_acts = []
        for elem in path_taken_s_a:
            #elem[0]-state
            target_acts.append(phy_policy[elem[0]])
            target_acts_stoch.append(phy_policy_stochastic[elem[0]])
            candidate_acts.append(opt_policy_learned[elem[0]])
        LL[traj_id] = {"IRL" : eu.log_likelihood(candidate_acts,target_acts,avg = avg),
                       "random" : avg * math.log((1/num_actions)) + (1-avg)* math.log((1/num_actions)) * len(target_acts),
                       "no_int" : eu.log_likelihood_no_act(target_acts_stoch,no_int_id,avg = avg),
                       "length" : len(target_acts)}
    return LL

def get_KL_divergence(phy_policy_stochastic, opt_policy_learned, significant_states = range(750)):
    opt_policy_learned = opt_policy_learned.query_Q_probs()
    phy_policy_stochastic = phy_policy_stochastic.query_Q_probs()
    no_int_policy = eu.gen_single_action_probability()
    random_policy = eu.gen_random_probability()

    vaso_only_random_policy = eu.gen_random_probability(random_action_range= range(5))
    iv_only_random_policy = eu.gen_random_probability(random_action_range= np.array(range(5)) * 5 )
    KL = {}
    for state in significant_states:
        phy_policy = phy_policy_stochastic[state]
        opt_policy = opt_policy_learned[state]
        KL[state] = {"IRL" : eu.KL_divergence(phy_policy,opt_policy),
                     "random" : eu.KL_divergence(phy_policy,random_policy),
                     "no_int_policy" : eu.KL_divergence(phy_policy,no_int_policy) ,
                     "vaso_only_random" : eu.KL_divergence(phy_policy,vaso_only_random_policy),
                     "iv_only_random" : eu.KL_divergence(phy_policy,iv_only_random_policy)
                    }
    return KL
