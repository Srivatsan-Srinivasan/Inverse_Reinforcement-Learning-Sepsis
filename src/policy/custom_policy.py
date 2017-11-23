import numpy as np
import pandas as pd
from constants import *
from policy.policy import GreedyPolicy, StochasticPolicy


def get_physician_policy(trajectories):
    '''
    get a clincian policy (greedy to their mode actions)
    '''
    # when tie, smallest index returned
    cols = ['s', 'a']
    df = pd.DataFrame(trajectories[:,1:3], columns=cols)
    groups_s = df.groupby(['s'])
    mode_actions = groups_s.agg(lambda x: x.value_counts().index[0])
    Q = np.eye(NUM_ACTIONS, dtype=np.float)[mode_actions]
    Q = Q.reshape(NUM_PURE_STATES, NUM_ACTIONS)
    pi = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q)
    return pi


def make_physician_stochastic_policy(trajectories, num_states, num_actions):
    '''
    another version that learns a stochastic policy
    based on the histogram of different actions given state
    TODO: not tested yet
    '''
    cols = ['s', 'a']
    df = pd.DataFrame(trajectories[:,1:3], columns=cols)
    groups_s = df.groupby(['s'])
    import pdb;pdb.set_trace()
    pi = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_table)
    return pi

