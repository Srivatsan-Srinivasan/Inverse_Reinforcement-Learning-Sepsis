import numpy as np
import pandas as pd
from constants import *
from policy.policy import GreedyPolicy


def get_physician_policy(trajectories):
    '''
    learn a policy where action taken by clinicians is used
    TODO: complete this
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


def get_physician_policy_2(trajectories):
    '''
    TODO: try another version: offline sampling sarsa?
    '''
    pass


def make_Q_clinician(trajectories, num_states, num_actions):
    '''
    another version that learns a stochastic policy
    based on the histogram of different actions given state
    TODO: complete this
    '''
    cols = ['s', 'a']
    df = pd.DataFrame(trajectories[:,1:3], columns=cols)
    groups_s = df.groupby(['s'])
    pi = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q_table)
    return pi

