import numpy as np
import pandas as pd
import itertools
# from constants import *

def make_initial_state_sampler(df):
    '''
    we only care about empirically observed initial states.
    '''
    initial_states = np.sort(df[df['bloc'] == 1]['state'].unique())
    def f():
        return np.random.choice(initial_states)
    return f

def make_initial_state_sampler_mock(task):
    task.reset()
    initial_states = task.observe()
    def f():
        return np.random.choice(initial_states)
    return f


def make_state_centroid_finder(df, columns=None):
    if columns is not None:
        df = df[columns]
    def f(state): # get_state(): a function to find centroid for a given state
        return df.iloc[state]
    return f

def make_state_centroid_finder_mock(task, NUM_STATES):
    '''
    Discretize 9x9 gridworld into 16 2x2 smaller grids/states + 2 addtional states (last row and last column).
    '''
    df_centroid_mock = np.zeros((NUM_STATES, 18))
    for state in range(NUM_STATES):
        if state in [0,1,9,10]:
            df_centroid_mock[state, 0] = 1
        elif state in [2,3,11,12]:
            df_centroid_mock[state, 1] = 1
        elif state in [4,5,13,14]:
            df_centroid_mock[state, 2] = 1
        elif state in [6,7,15,16]:
            df_centroid_mock[state, 3] = 1
        elif state in [18,19,27,28]:
            df_centroid_mock[state, 4] = 1
        elif state in [20,21,29,30]:
            df_centroid_mock[state, 5] = 1
        elif state in [22,23,31,32]:
            df_centroid_mock[state, 6] = 1
        elif state in [24,25,33,34]:
            df_centroid_mock[state, 7] = 1
        elif state in [36,37,45,46]:
            df_centroid_mock[state, 8] = 1
        elif state in [38,39,47,48]:
            df_centroid_mock[state, 9] = 1
        elif state in [40,41,49,50]:
            df_centroid_mock[state, 10] = 1
        elif state in [42,43,51,52]:
            df_centroid_mock[state, 11] = 1
        elif state in [54,55,63,64]:
            df_centroid_mock[state, 12] = 1
        elif state in [56,57,65,66]:
            df_centroid_mock[state, 13] = 1
        elif state in [58,59,67,68]:
            df_centroid_mock[state, 14] = 1
        elif state in [60,61,69,70]:
            df_centroid_mock[state, 15] = 1
        elif state in [8,17, 26,35,44,53,62,71]:
            df_centroid_mock[state, 16] = 1
        elif state in [72,73,74,75,76,77,78,79,80]:
            df_centroid_mock[state, 17] = 1
        # if task.is_terminal( state ):
        #     df_centroid_mock[state, 18] = 1
        # if task.is_wall(state):
        #     df_centroid_mock[state,19] = 1

    def f(state): # get_state(): a function to find centroid for a given state
        return df_centroid_mock[state]
    return f

def estimate_feature_expectation(task, transition_matrix, sample_initial_state, get_state, phi, pi,
                                 gamma=0.99, num_trajectories=100):
    max_iter = 500
    s = sample_initial_state()
    s_cent = get_state(s)
    mu = np.zeros(phi(s_cent).shape)
    

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
            
            if task.is_terminal( new_s ):
                break
            s = new_s
            s_cent = get_state(new_s)
      
    mu = (1.0 * mu) / num_trajectories
    return mu



def phi(state_centroid_mock):
    '''
    state: centroid values whose dimension is {num_features}
    phi: must apply decision rule (=indicator function)

    returs: binary matrix of R^{num_features}
    '''
    # TODO: implement this
    phi_s = state_centroid_mock
    return phi_s


def make_reward_computer(W, get_state, phi):
    def compute_reward(state):
        # if task.is_terminal(state):
        #     # special case of terminal states
        #     # either 1 or -1
        #     num_features = W.shape[0]
        #     return compute_terminal_state_reward(state, num_features)
        s_cent = get_state(state)
        return np.dot(W, phi(s_cent))
    return compute_reward

# def estimate_v_pi_tilda(W, mu, sample_initial_state, sample_size=100):
#     v_pi_tilda = np.dot(W, mu)
#     # remove two terminal_states
#     v_pi_tilda = v_pi_tilda[:v_pi_tilda.shape[0] - NUM_TERMINAL_STATES]
#     v_pi_tilda_est = 0.0
#     for _ in range(sample_size):
#         s_0 = sample_initial_state()
#         v_pi_tilda_est += v_pi_tilda[s_0]
#     return v_pi_tilda_est/sample_size
