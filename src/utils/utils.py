import os
import numpy as np
import numba as nb
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans 
from sklearn import preprocessing 
from constants import *

def check_numerical_categorical(all_cols, categorical_cols, numerical_cols):
    check1 = (set(numerical_cols) - set(all_cols))
    #print(check1)
    check2 = (set(categorical_cols) - set(all_cols))
    #print(check2)
    check3 = (set(all_cols) - set(categorical_cols) - set(numerical_cols))
    #print(check3)
    return (check1 | check2 | check3) == set(ETC)


def load_data():
    # TODO: accept filepath to get train/test/vali
    num_states = NUM_STATES - NUM_TERMINAL_STATES
    if os.path.isfile(CLEANSED_DATA_FILEPATH):
        df = _load_data(FILEPATH)
        df_cleansed = _load_data(CLEANSED_DATA_FILEPATH)
        df_centroids = _load_data(CENTROIDS_DATA_FILEPATH)
    else:
        df = _load_data(FILEPATH)
        df_corrected = correct_data(df)
        df_norm = normalize_data(df_corrected)
        X, mu, y = separate_X_mu_y(df_norm, ALL_VALUES)
        X_to_cluster = X.drop(COLS_NOT_FOR_CLUSTERING, axis=1)
        X_centroids, X_clustered = clustering(X_to_cluster, k=num_states, batch_size=500)
        X['state'] = pd.Series(X_clustered)
        df_cleansed = pd.concat([X, mu, y], axis=1)
        df_centroids = pd.DataFrame(X_centroids, columns=X_to_cluster.columns)
        
        df_cleansed.to_csv(CLEANSED_DATA_FILEPATH, index=False)
        df_centroids.to_csv(CENTROIDS_DATA_FILEPATH, index=False)

    return df, df_cleansed, df_centroids

def _load_data(path):
    df = pd.read_csv(path)
    return df


def _load_data2(path):
    '''
    curious about the motivation of this fuction
    '''
    df = pd.read_csv(path)
    cols = df.column
    valid_cols = list(set(INTEGER_COLS) & set(cols))
    df[valid_cols].astype(np.int)
    return df


def extract_trajectories(df, num_states):
    if os.path.isfile(TRAJECTORIES_FILEPATH):
        trajectories = np.load(TRAJECTORIES_FILEPATH)
    else:
        print('extract trajectories')
        trajectories = _extract_trajectories(df, num_states)
        np.save(TRAJECTORIES_FILEPATH, trajectories)
    return trajectories


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
   

def normalize_data(df):
    # divide cols: numerical, categorical, text data
    # logarithimic scale 
    df[COLS_TO_BE_NORMALIZED] -= np.mean(df[COLS_TO_BE_NORMALIZED], axis=0)
    df[COLS_TO_BE_NORMALIZED] /= np.std(df[COLS_TO_BE_NORMALIZED], axis=0)
    return df


def correct_data(df):
    # the logic is hard-coded. could be fixed...
    # TODO: vectorize this
    df['sedation'] = df['sedation'].clip(0.0)

    for i, c in enumerate(COLS_TO_BE_LOGGED):
        # ideally correct missing data or obviously wrong values
        stats = df[c].describe(percentiles=[.01, 0.99])
        min_val = stats['1%']
        max_val = stats['99%']
        df[c] = df[c].clip(min_val, max_val)
    for i, c in enumerate(COLS_TO_BE_LOGGED):
        # k means clustering assumes vars are normally distributed
        # to control the effect of outliers or a certain var
        # we artificially make the vars look more normal
        df[c] = np.log(df[c])
        finite_min = df[c][np.isfinite(df[c])].min()
        df[c] = df[c].clip(finite_min)
        stats = df[c].describe()
        #sns.distplot(df[c], color=palette[i])
        #plt.show()
    return df


def separate_X_mu_y(df, cols=None):
    mu = df[INTERVENTIONS]
    # TODO: fix the error here
    mu['action'], tev_bin_edges, vaso_bin_edges  = discretize_actions(mu['input_4hourly_tev'], mu['median_dose_vaso'])
    y = df[OUTCOMES]
    if cols is None:
        default_cols = set(ALL_VALUES) - set(OUTCOMES)
        X = df[list(default_cols)]
    else:
        observation_cols = set(cols) - set(OUTCOMES) - set(INTERVENTIONS)
        X = df[list(observation_cols)]
    return X, mu, y


def apply_pca(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    # explained variance of the first column is already 99%
    X_pca = pd.DataFrame(X_pca, columns=list('AB'))
    return X_pca


def clustering(X, k=2000, batch_size=100):
    # pick only numerical columns that make sense
    mbk = MiniBatchKMeans(n_clusters=k, batch_size=batch_size, init_size=k*3)
    mbk.fit(X)
    X_centroids = mbk.cluster_centers_
    X_clustered = mbk.predict(X)
    return X_centroids, X_clustered


def discretize_actions(
        input_4hourly__sequence__continuous,
        median_dose_vaso__sequence__continuous,
        bins_num = 5):
    # IV fluids discretization
    input_4hourly__sequence__continuous__no_zeros = input_4hourly__sequence__continuous[ \
        input_4hourly__sequence__continuous > 0]
    input_4hourly__sequence__discretized__no_zeros, input_4hourly__bin_bounds = \
        pd.qcut( input_4hourly__sequence__continuous__no_zeros, \
                 bins_num - 1, labels = False, retbins = True)
    input_4hourly__sequence__discretized = \
        (input_4hourly__sequence__continuous > 0).astype(int)
    input_4hourly__sequence__discretized[ input_4hourly__sequence__discretized == 1 ] = \
        input_4hourly__sequence__discretized__no_zeros + 1
        
    # Vaopressors discretization
    median_dose_vaso__sequence__continuous__no_zeros = median_dose_vaso__sequence__continuous[ \
        median_dose_vaso__sequence__continuous > 0]
    median_dose_vaso__sequence__discretized__no_zeros, median_dose_vaso__bin_bounds = \
        pd.qcut( median_dose_vaso__sequence__continuous__no_zeros, \
                 bins_num - 1, labels = False, retbins = True)
    median_dose_vaso__sequence__discretized = \
        (median_dose_vaso__sequence__continuous > 0).astype(int)
    median_dose_vaso__sequence__discretized[ median_dose_vaso__sequence__discretized == 1 ] = \
        median_dose_vaso__sequence__discretized__no_zeros + 1
        
    # Combine both actions discretizations
    actions_sequence = median_dose_vaso__sequence__discretized * bins_num + \
        input_4hourly__sequence__discretized
    
    # Calculate for IV fluids quartiles the median dose given in that quartile
    input_4hourly__conversion_from_binned_to_continuous = np.zeros(bins_num)
    for bin_ind in range(1, bins_num):
        input_4hourly__conversion_from_binned_to_continuous[bin_ind] = \
        np.median(input_4hourly__sequence__continuous__no_zeros[ \
                  input_4hourly__sequence__discretized__no_zeros == bin_ind - 1] )
    
    # Calculate for vasopressors quartiles the median dose given in that quartile
    median_dose_vaso__conversion_from_binned_to_continuous = np.zeros(bins_num)
    for bin_ind in range(1, bins_num):
        median_dose_vaso__conversion_from_binned_to_continuous[bin_ind] = \
        np.median(median_dose_vaso__sequence__continuous__no_zeros[ \
                  median_dose_vaso__sequence__discretized__no_zeros == bin_ind - 1] )
    
    return actions_sequence, \
        input_4hourly__conversion_from_binned_to_continuous, \
        median_dose_vaso__conversion_from_binned_to_continuous

def is_terminal_state(s):
    return s >= (NUM_STATES - NUM_TERMINAL_STATES)

def compute_terminal_state_reward(s, num_features):
    if s == TERMINAL_STATE_ALIVE:
        return np.sqrt(num_features)
    elif s == TERMINAL_STATE_DEAD:
        return -np.sqrt(num_features)
    else:
        raise Exception('not recognizing this terminal state: '.foramt(s))
