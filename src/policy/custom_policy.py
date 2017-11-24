import numpy as np
import pandas as pd
from constants import *
from policy.policy import GreedyPolicy, StochasticPolicy


def get_physician_policy(trajectories, is_stochastic=False):
    '''
    get a clincian policy (greedy to their mode actions)
    '''
    # when tie, smallest index returned
    df = pd.DataFrame(trajectories[:,1:3], columns=['s', 'a'])
    trajectories = trajectories.astype(np.int)
    groups_s = df.groupby(['s'])
    if is_stochastic:
        Q = np.zeros((NUM_STATES, NUM_ACTIONS))
        action_dist = df.groupby(['s'])['a'].apply(lambda x : x.value_counts())
        idx = np.array([[x[0], x[1]] for x in action_dist.index.values])
        Q[idx[:,0], idx[:,1]] = action_dist
        pi = StochasticPolicy(NUM_STATES, NUM_ACTIONS, Q)
    else:
        mode_actions = df.groupby(['s']).agg(lambda x: x.value_counts().index[0])['a']
        Q = np.eye(NUM_ACTIONS, dtype=np.float)[mode_actions]
        Q = np.vstack([Q, np.zeros((NUM_TERMINAL_STATES, NUM_ACTIONS))])
        pi = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q)
    return pi


