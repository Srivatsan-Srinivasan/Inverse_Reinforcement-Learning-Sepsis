import numpy as np
import pandas as pd
import os
from constants import *
from utils.utils import *

def make_mdp(df_cleansed, num_states, num_actions):
    '''
    build states by running k-means clustering
    Note: we exclude nominal categorical columns from clustering.
    the columns to be excluded are: chartime, icustyaid, bloc
    '''
    if os.path.isfile(TRAJECTORIES_FILEPATH):
        trajectories = np.load(TRAJECTORIES_FILEPATH)
    else:
        print('extract trajectories')
        trajectories = _extract_trajectories(df_cleansed, num_states)
        np.save(TRAJECTORIES_FILEPATH, trajectories)
    
    if os.path.isfile(TRANSITION_MATRIX_FILEPATH) and \
            os.path.isfile(REWARD_MATRIX_FILEPATH):
        transition_matrix = np.load(TRANSITION_MATRIX_FILEPATH)
        reward_matrix = np.load(REWARD_MATRIX_FILEPATH)
    else:
        print('making mdp')
        transition_matrix, reward_matrix = _make_mdp(trajectories, num_states, num_actions)
        np.save(TRANSITION_MATRIX_FILEPATH, transition_matrix)
        np.save(REWARD_MATRIX_FILEPATH, reward_matrix)
    return transition_matrix, reward_matrix


def _make_mdp(trajectories, num_states, num_actions):
    transition_matrix = np.zeros((num_states, num_actions, num_states))
    reward_matrix = np.zeros((num_states + num_terminal_states, num_actions))
    # stochastic world: 1% of uncertainty in transition
    eps = 1e-2
    TRANSITION_PROB_UNVISITED_SAS = eps / num_states
    # if (s, a) never observed, we naively assume uniform transition
    TRANSITION_PROB_UNVISITED_SA = 1.0 / num_states
    REWARD_UNVISITED_SA = 0.0

    # create dataframe for easy tallying
    cols = ['s', 'a', 'r', 'new_s']
    df = pd.DataFrame(trajectories[:,1:], columns=cols)
    groups_sas = df.groupby(['s', 'a', 'new_s'])
    groups_sa = df.groupby(['s', 'a'])
    avg_reward_sa = groups_sa['r'].mean()
    transition_count_sa = groups_sa.size()
    transition_count_sas = groups_sas.size()
    
    # TODO: vectorize this
    i = 0
    print('this is a loop of length', num_states**2 * num_actions)
    for s in range(num_states):
        if s == (num_states - 1) or s == (num_states - 2):
            # TODO: fix this hardcoding
            # if terminal states, must be absorbing
            transition_matrix[s, :, s] = 1.0
            continue
        for a in range(num_actions):
            # store empirical reward
            if (s, a) in avg_reward_sa:
                reward_matrix[s, a] = avg_reward_sa[(s, a)]
            else:
                reward_matrix[s, a] = REWARD_UNVISITED_SA
            # store empirical transitions
            if (s, a) in transition_count_sa:
                # give small trans. prob to every next state
                transition_matrix[s, a, :] = TRANSITION_PROB_UNVISITED_SAS
                num_sa = transition_count_sa[(s, a)]
                for new_s in range(num_states):
                    i+=1
                    if i % 10000 == 0:
                        print('patience is a virtue. state: {}'.format(s, a))
                    if (s, a, new_s) in transition_count_sas:
                        num_sas = transition_count_sas[(s, a, new_s)]
                        transition_matrix[s, a, new_s] += (1 - eps)*num_sas / num_sa
            else:
                # if (s, a) never observed, we naively assume uniform transition
                transition_matrix[s, a, :] = TRANSITION_PROB_UNVISITED_SA

    return transition_matrix, reward_matrix

    
def _extract_trajectories(df, num_states):
    '''
    a few strong assumptions are made here.
    1. we consider those who died in 90 days but not in hopsital to have the same status as alive. hence we give reward of one. Worry not. we can change back. this assumption was to be made to account for uncertainty in the cause of dealth after leaving the hospital
    '''
    cols = ['icustayid', 's', 'a', 'r', 'new_s']
    df = df.sort_values(['icustayid', 'bloc'])
    groups = df.groupby('icustayid')
    trajectories = pd.DataFrame(np.zeros((df.shape[0], len(cols))), columns=cols)
    trajectories.loc[:, 'icustayid'] = df['icustayid']
    trajectories.loc[:, 's'] = df['state']
    trajectories.loc[:, 'a'] = df['action']

    # reward function
    DEFAULT_REWARD = 0
    trajectories.loc[:, 'r'] = DEFAULT_REWARD
    terminal_steps = groups.tail(1).index
    is_terminal = df.isin(df.iloc[terminal_steps, :]).iloc[:, 0]
    died_in_hosp = df[OUTCOMES[0]] == 1
    died_in_90d = df[OUTCOMES[1]] == 1
    # reward for those who survived (order matters)
    trajectories.loc[is_terminal, 'r'] = 1
    trajectories.loc[is_terminal & died_in_hosp, 'r'] = -1

    # TODO: vectorize this
    new_s = pd.Series([])
    for name, g in groups:
        # add three imaginary states
        # to simplify, we use died_in_hosp_only
        if np.any(g['died_in_hosp'] == 1):
            terminal_marker = TERMINAL_STATE_DEAD
        else:
            # survived
            terminal_marker = TERMINAL_STATE_ALIVE
        new_s_sequence = g['state'].shift(-1)
        new_s_sequence.iloc[-1] = terminal_marker
        new_s = pd.concat([new_s, new_s_sequence])
    trajectories.loc[:, 'new_s'] = new_s.astype(np.int)

    return trajectories.as_matrix()
    
