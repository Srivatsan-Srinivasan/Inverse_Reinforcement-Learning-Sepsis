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
    Q = np.zeros((NUM_PURE_STATES, NUM_ACTIONS))
    action_dist = df.groupby(['s'])['a'].apply(lambda x : x.value_counts())
    idx = np.array([[x[0], x[1]] for x in action_dist.index.values])
    Q[idx[:,0], idx[:,1]] = action_dist
    if is_stochastic:
        pi = StochasticPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q)
    else:
        pi = GreedyPolicy(NUM_PURE_STATES, NUM_ACTIONS, Q)
    return pi


