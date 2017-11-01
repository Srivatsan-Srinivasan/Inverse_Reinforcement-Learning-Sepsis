import numpy as np
import pandas as np
import itertools
from constants import *
from utils.utils import is_terminal_state, compute_terminal_state_reward

def make_initial_state_sampler(df):
    '''
    we only care about empirically observed initial states.
    '''
    initial_states = np.sort(df[df['bloc'] == 1]['state'].unique())
    def f():
        return np.random.choice(initial_states)
    return f


def make_state_centroid_finder(df, columns=None):
    if columns is not None:
        df = df[columns]
    def f(state):
        return df.iloc[state]
    return f


def estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, pi, gamma=0.99, num_trajectories=100):
    # TODO: get_state is ugly. fix this
    s = sample_initial_state()
    s_cent = get_state(s)
    mu = np.zeros(phi(s_cent).shape)

    for i in range(num_trajectories):
        s = sample_initial_state()
        s_cent = get_state(s)
        for t in itertools.count():
            # accumulate phi(s) over trajectories
            mu += gamma**t * phi(s)
            # sample next action
            probs = pi.query_Q_probs(s)
            chosen_a = np.random.choice(np.arange(len(probs)), p=probs)
            # sample next state
            # need to renomralize so sum(probs) < 1
            probs = transition_matrix[s, chosen_a, :]
            probs /= np.sum(probs)
            new_s = np.random.choice(np.arange(len(probs)), p=probs)
            
            if is_terminal_state(new_s):
                # there's no phi(terminal_state)
                break
            s = new_s
            s_cent = get_state(new_s)
      
    mu = (1.0 * mu) / num_trajectories
    return mu


def dummy_phi(states):
    return states


def phi(state):
    '''
    state: centroid values whose dimension is {num_features}
    phi: must apply decision rule (=indicator function)

    returs: binary matrix of R^{num_features}
    '''
    # TODO: implement this
    return np.uint8(state > 0)


def estimate_v_pi(W, mu):
    return np.dot(W, mu)


def make_reward_computer(W, get_state):
    def compute_reward(state):
        if is_terminal_state(state):
            # special case of terminal states
            # either 1 or -1
            return compute_terminal_state_reward(state)
        s_cent = get_state(state)
        return np.dot(W, phi(s_cent))
    return compute_reward

def estimate_v_pi_tilda(v_pi_tilda, sample_initial_state, W, sample_size=100):
    # remove two terminal_states
    v_pi_tilda = v_pi_tilda[:v_pi_tilda.shape[0] - NUM_TERMINAL_STATES]
    v_pi_tilda_est = 0.0
    for _ in range(sample_size):
        s_0 = sample_initial_state()
        v_pi_tilda_est += v_pi_tilda[s_0]
    return v_pi_tilda_est/sample_size
