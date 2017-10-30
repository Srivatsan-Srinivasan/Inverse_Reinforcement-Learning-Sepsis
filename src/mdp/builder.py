import numpy as np
import pandas as pd
import os
from constants import *
from utils.utils import load_data

def make_mdp(num_states, num_actions):
    # build mdp
    if os.path.isfile(CLEANSED_DATA_FILEPATH):
        df_cleansed = load_data(CLEANSED_DATA_FILEPATH)
    else:
        df = load_data(FILEPATH)
        df_corrected = correct_data(df)
        df_norm = normalize_data(df_corrected)
        X, mu, y = separate_X_mu_y(df_norm, ALL_VALUES)
        X_clustered = clustering(X, k=num_states, batch_size=100)
        X['state_cluster'] = pd.Series(X_clustered)
        df_cleansed = pd.concat([X, mu, y], axis=1)
        df_cleansed.to_csv(CLEANSED_DATA_FILEPATH, index=False)

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
    return df_cleansed, transition_matrix, reward_matrix


def _make_mdp(trajectories, num_states, num_actions):
    # TODO: fix this hard coding
    num_terminal_states = 3
    transition_matrix = np.zeros((num_states + num_terminal_states, num_actions, num_states + num_terminal_states))
    reward_matrix = np.zeros((num_states + num_terminal_states, num_actions))
    TRANSITION_PROB_UNVISITED_SAS = 0.0
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
    # but everything is O(1) inside the loop so it's O(n^2m)
    # TODO: consider mark transition to the imaginary terminal states
    # to the prob of 1.0. this may be undesirable consequences
    i = 0
    print('this is a loop of length', num_states**2 * num_actions)
    for s in range(num_states):
        for a in range(num_actions):
            # handle reward
            if (s, a) in avg_reward_sa:
                reward_matrix[s, a] = avg_reward_sa[(s, a)]
            else:
                reward_matrix[s, a] = REWARD_UNVISITED_SA
            # handle transitions
            if (s, a) in transition_count_sa:
                num_sa = transition_count_sa[(s, a)]
                for new_s in range(num_states):
                    i+=1
                    if i % 10000 == 0:
                        print('i am doing fine, progress:', s, a, new_s)
                    if (s, a, new_s) in transition_count_sas:
                        num_sas = transition_count_sas[(s, a, new_s)]
                        transition_matrix[s, a, new_s] = num_sas / num_sa
                    else:
                        transition_matrix[s, a, new_s] = TRANSITION_PROB_UNVISITED_SAS
            else:
                transition_matrix[s, a, :] = TRANSITION_PROB_UNVISITED_SAS

    return transition_matrix, reward_matrix

    
def _extract_trajectories(df, num_states):
    # patient id, s, a, r, new_s
    cols = ['icustayid', 's', 'a', 'r', 'new_s']
    df = df.sort_values(['icustayid', 'bloc'])
    groups = df.groupby('icustayid')
    DEFAULT_REWARD = 0
    trajectories = pd.DataFrame(np.zeros((df.shape[0], len(cols))), columns=cols)
    trajectories.loc[:, 'icustayid'] = df['icustayid']
    trajectories.loc[:, 's'] = df['state_cluster']
    trajectories.loc[:, 'a'] = df['action_bin']

    # TODO: fix so that the terminal state does not get reward
    # reward function
    trajectories.loc[:, 'r'] = DEFAULT_REWARD
    terminal_steps = groups.tail(1).index
    is_terminal = df.isin(df.iloc[terminal_steps, :]).iloc[:, 0]
    died_in_hosp = df[OUTCOMES[0]] == 1
    died_in_90d = df[OUTCOMES[1]] == 1
    # reward for those who survived (order matters)
    trajectories.loc[is_terminal, 'r'] = 20
    trajectories.loc[is_terminal & died_in_hosp, 'r'] = -20
    trajectories.loc[is_terminal & died_in_90d, 'r']  = -10
    #trajectories.loc[terminal_steps, 'r'] = modify_reward(df.loc[terminal_steps, OUTCOMES])

    # TODO: vectorize this
    new_s = pd.Series([])
    for name, g in groups:
        # TODO: fix the last terminal step
        new_s_sequence = g['state_cluster'].shift(-1)
        # use of the same terminal_marker does not make sense
        # as different patients exit mdp with varying conditions
        if np.any(g['died_in_hosp'] == 1):
            terminal_marker = num_states
        elif np.any(g['mortality_90d'] == 1):
            terminal_marker = num_states + 1
        else:
            # survived
            terminal_marker = num_states + 2
        new_s_sequence.iloc[-1] = terminal_marker
        new_s = pd.concat([new_s, new_s_sequence])
    trajectories.loc[:, 'new_s'] = new_s.astype(np.int)
    # return as numpy 2d array
    return trajectories.as_matrix()
    
