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
    return cross_entropy(p,q) + single_entropy(p)

def cross_entropy(p,q):
    return entropy(p,q)

def single_entropy(p):
    return entropy(p)

#Mutual information score
def mis(p,q):
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
    for act in target_actions:
        out += math.log(prob_actions[act])






    
    


