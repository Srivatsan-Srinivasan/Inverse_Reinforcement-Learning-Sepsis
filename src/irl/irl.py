import numpy as np
import pandas as np
from constants import *

# columns that are binary:
def find_binary_columns():
    df = load_data(FILEPATH)
    variables_to_use = []
    for i in range(df.shape[1]):
        if len(df.iloc[:,i].unique())==2:
            variables_to_use.append(i)
    #exclude last two columns: died_in_hosp and mortality_90d
    variables_to_use = variables_to_use[:-2]
    return variables_to_use


# state basis function
def phi(centroid, state, variables_to_use):
    phi_st = centroid[state, variables_to_use]
    return phi_st


# sampling heuristic trajectories
def sampling_trajectories(transition_matrix, policy, m, state_count):
    absorption_states = [state_count, state_count+1, state_count+2]
    
    keys = range(m)
    sample_trajectories = dict.fromkeys(keys, None)
    
    for i in keys:
        sample_trajectories[i] = []
        #start from a random state
        state = int(np.random.choice(range(state_count)))
        sample_trajectories[i].append(state)
        while state not in absorption_states:
            action = int(policy[state])
            probs = transition_matrix[state,action,:]
            next_state = int(np.random.choice(np.arange(state_count+len(absorption_states)), p=probs))
            sample_trajectories[i].append(next_state)
            state = np.copy(next_state)
    return sample_trajectories

    
# function to calculate mu
def feature_expectation(sample_trajectories, gamma):
    mu = np.zeros((variable_count))
    
    # loop over all trajectories
    for i in range(m):
        trajectory = sample_trajectories[i]
        # loop over all states in that trajectory
        t = 0
        for state in trajectory:
            phi_st = phi(centroid, state, variables_to_use)
            mu += gamma**t*phi_st
            t+=1
    mu = 1/float(m)*mu    
    return mu


def reward(w, centroid, variables_to_use):
    R = w*centroid[:, variables_to_use]
    return R

