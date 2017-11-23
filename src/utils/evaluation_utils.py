# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:20:13 2017

@author: SrivatsanPC
"""
from sklearn.metrics import mutual_info_score as mis
from scipy.stats import entropy
import numpy as np
import math

#p,q are distributions - p is base distribution and q is approximate
def KL_divergence(p,q):
    return entropy(p,q) 

def cross_entropy(p,q):
    return entropy(p,q) + entropy(p) #KL-divergence + entropy

def single_entropy(p):
    return entropy(p)

#Mutual information score
def mis_score(p,q):
    return mis(p,q)

def gen_random_probability(num_actions = 25):
    out = np.array([1/num_actions for i in range(num_actions)])
    assert(sum(out) == 1 and len(out) == num_actions)
    return out

def gen_single_action_probability(num_actions = 25, action_id = 0):
    out = np.zeros(num_actions)
    out[action_id] = 1
    assert(sum(out) == 1 and len(out) == num_actions)
    return out

def log_likelihood(prob_actions, target_actions):
    out = 0
    #SANITY checks.
    assert(len(prob_actions) == len(target_actions))
    assert(len(prob_actions[0]) == 25)
    
    for i in range(len(target_actions)):
        target_act = target_actions[i]
        candidate_act_prob = prob_actions[i]
        out += math.log(candidate_act_prob[target_act])
    return out

def log_likelihood_no_act(target_acts_stoch, no_int_id):
    sum([math.log(i[no_int_id] + 1e-3) for i in target_acts_stoch ])


def test_code():
    a = [0.1,0.2,0.3,0.4]
    b = [0.1,0.2,0.3,0.4]
    c = [0.11, 0.21, 0.29, 0.39]
    d = [0.4,0.3,0.2,0.1]
    e = [0.25,0.25,0.25,0.25]
    
    print("KL(",a,",",b,") : ",KL_divergence(a,b))
    print("KL(",a,",",c,") : ",KL_divergence(a,c))
    print("KL(",a,",",d,") : ",KL_divergence(a,d))
    print("KL(",a,",",e,") : ",KL_divergence(a,e))
    
    print("CE(",a,",",b,") : ",cross_entropy(a,b))
    print("CE(",a,",",c,") : ",cross_entropy(a,c))
    print("CE(",a,",",d,") : ",cross_entropy(a,d))
    print("CE(",a,",",e,") : ",cross_entropy(a,e))
 


test_code()
    
    


