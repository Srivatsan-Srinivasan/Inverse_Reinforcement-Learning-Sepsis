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

def estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, phi, pi,
                                 gamma=0.99, num_trajectories=700):
    '''
    estimate mu_pi and v_pi with monte carlo simulation
    '''
    max_iter = 500
    # TODO: get_state is ugly. fix this
    s = sample_initial_state()
    s_cent = get_state(s)
    mu = np.zeros(phi(s_cent).shape)
    v_sum = 0.0
    
    for i in range(num_trajectories):
        s = sample_initial_state()
        s_cent = get_state(s)
        for t in itertools.count():
            if t > max_iter:
                break
            # accumulate phi(s) over trajectories
            mu += gamma**t * phi(s_cent)
            # sample next action
            probs = pi.query_Q_probs(s)
            chosen_a = np.random.choice(np.arange(len(probs)), p=probs)
            # sample next state
            # need to renomralize so sum(probs) < 1
            probs = np.copy(transition_matrix[s, chosen_a, :])
            probs /= np.sum(probs)
            new_s = np.random.choice(np.arange(len(probs)), p=probs)
            
            if is_terminal_state(new_s):
                # there's no phi(terminal_state)
                # in practice, non-zero rewars for terminal states
                num_features = mu.shape[0]
                v_sum += gamma** t * compute_terminal_state_reward(new_s, num_features)
                break
            s = new_s
            s_cent = get_state(new_s)
      
    mu = (1.0 * mu) / num_trajectories
    v =  v_sum / num_trajectories
    return mu, v


def make_phi(df_centroids):
    # median values for each centroid
    stats = df_centroids.describe()
    #take median
    median_state = stats.loc['50%']
    def phi(state):
        '''
        state: centroid values whose dimension is {num_features}
        phi: must apply decision rule (=indicator function)

        returs: binary matrix of R^{num_features}
        '''
        # TODO: implement this
        phi_s = np.array((state > median_state).astype(np.int))
        return phi_s
    return phi


def make_reward_computer(W, get_state, phi):
    def compute_reward(state):
        if is_terminal_state(state):
            # special case of terminal states
            # either 1 or -1
            num_features = W.shape[0]
            return compute_terminal_state_reward(state, num_features)
        s_cent = get_state(state)
        return np.dot(W, phi(s_cent))
    return compute_reward

def estimate_v_pi_tilda(W, mu, sample_initial_state, sample_size=100):
    # this does not work. don't use this for now.
    v_pi_tilda = np.dot(W, mu)
    # remove two terminal_states
    v_pi_tilda = v_pi_tilda[:v_pi_tilda.shape[0] - NUM_TERMINAL_STATES]
    v_pi_tilda_est = 0.0
    # TODO: vectorize this
    for _ in range(sample_size):
        s_0 = sample_initial_state()
        v_pi_tilda_est += v_pi_tilda[s_0]
    return v_pi_tilda_est / sample_size
