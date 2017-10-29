import matplotlib.pyplot as plt
import seaborn as sns
import os

from policy.policy import GreedyPolicy
from mdp.dynamic_programming import iterate_policy, evaluate_policy
from utils.utils import * 
from data_hw2.constants import *
from policy.policy import GreedyPolicy

import seaborn as sns
sns.set(palette="husl", style="dark")
palette = sns.color_palette("husl", 70)

'''
TODO
[ ] increase k to 2000
[ ] extract clinician policy

'''

def make_mdp(trajectories, num_states, num_actions):
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

    
def extract_trajectories(df, num_states):
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
    
def modify_reward(df):
    # currently not using this
    import pdb;pdb.set_trace()
    df_ret = pd.DataFrame()
    died_in_hosp = df[OUTCOMES[0]] == 1
    died_in_90d = df[OUTCOMES[1]] == 1
    df.loc[died_in_hosp, 'r'] -= 20
    df.loc[died_in_90d, 'r'] -= 10
    return df['r']

def solve_mdp(transition_matrix, reward_matrix):
    Q = np.zeros(reward_matrix.shape)
    Q_star = iterate_policy(Q, transition_matrix, reward_matrix)
    return Q_star
    

def evaluate(policies, transition_matrix, reward_matrix):
    V_pis = []
    for p in policies:
        V_pi = evaluate_policy(p, transition_matrix, reward_matrix)
        V_pis.append(V_pi)
    return V_pis

def sample_patients(df, sample_size=20000):
    # TODO: Fix this
    samples = np.random.choice(df['icustayid'], sample_size)
    import pdb;pdb.set_trace()
    return df['icustayid', samples]

def make_Q_random(num_states, num_actions):
    Q = np.zeros((num_states, num_actions))
    uniform_Q = (1.0 / num_actions)
    Q.fill(uniform_Q)
    return Q

def make_Q_nointerv(num_states, num_actions):
    Q = np.zeros((num_states, num_actions))
    NOINTERV_ACTION_IDX = 0
    # this is not prob
    Q[:, NOINTERV_ACTION_IDX] = 1.0
    return Q

def make_Q_clinician(trajectories, num_states, num_actions):
    # TODO: reimplement with offline sampling SARSA
    # check objective 1 in matthiue's slides
    Q = np.zeros((num_states, num_actions))
    # return mode of action taken by clinicians given state
    cols = ['s', 'a']
    df = pd.DataFrame(trajectories[:,1:3], columns=cols)
    count_sa = df.groupby(cols).size()
    for s in range(num_states):
        for a in range(num_actions):
            # TODO: fix this loop
            if (s, a) in count_sa:
                Q[s, a] = count_sa[(s, a)]
    return Q

if __name__ == '__main__':
    # build mdp
    num_states = 750
    num_actions = 25

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
        trajectories = extract_trajectories(df_cleansed, num_states)
        np.save(TRAJECTORIES_FILEPATH, trajectories)
    
    if os.path.isfile(TRANSITION_MATRIX_FILEPATH) and \
            os.path.isfile(REWARD_MATRIX_FILEPATH):
        transition_matrix = np.load(TRANSITION_MATRIX_FILEPATH)
        reward_matrix = np.load(REWARD_MATRIX_FILEPATH)
    else:
        print('making mdp')
        transition_matrix, reward_matrix = make_mdp(trajectories, num_states, num_actions)
        np.save(TRANSITION_MATRIX_FILEPATH, transition_matrix)
        np.save(REWARD_MATRIX_FILEPATH, reward_matrix)

    print('eval clinician policy')
    if os.path.isfile(Q_CLINICIAN_FILEPATH):
        Q_clinician = np.load(Q_CLINICIAN_FILEPATH)
        vf_clinician = np.load(VF_CLINICIAN_FILEPATH)
    else:
        Q_clinician = make_Q_clinician(trajectories, num_states, num_actions)
        np.save(Q_CLINICIAN_FILEPATH, Q_clinician)
        vf_clinician = evaluate_policy(Q_clinician, transition_matrix, reward_matrix)
        np.save(VF_CLINICIAN_FILEPATH, vf_clinician)

    print('eval random policy')
    if os.path.isfile(VF_RANDOM_FILEPATH):
        Q_random = np.load('data_hw2/q_random.npy')
        vf_random = np.load(VF_RANDOM_FILEPATH)
    else:
        Q_random = make_Q_random(num_states, num_actions)
        np.save('data_hw2/q_random.npy', Q_random)
        vf_random = evaluate_policy(Q_random, transition_matrix, reward_matrix)
        np.save('data_hw2/vf_random.npy', vf_random)
    
    print('eval no interv policy')
    if os.path.isfile('data_hw2/vf_nointerv.npy'):
        Q_nointerv = np.load('data_hw2/q_nointerv.npy')
        vf_nointerv = np.load('data_hw2/vf_nointerv.npy')
    else:
        # eval no intervention action
        Q_nointerv = make_Q_nointerv(num_states, num_actions)
        np.save('data_hw2/q_nointerv.npy', Q_nointerv)
        vf_nointerv = evaluate_policy(Q_nointerv, transition_matrix, reward_matrix)
        np.save('data_hw2/vf_nointerv.npy', vf_nointerv)

    print('eval optimal policy')
    if os.path.isfile(Q_STAR_FILEPATH):
        Q_star = np.load(Q_STAR_FILEPATH)
        vf_star = np.load('data_hw2/vf_star.npy')
    else:
        Q_star = solve_mdp(transition_matrix, reward_matrix)
        np.save(Q_STAR_FILEPATH, Q_star)
        vf_star = evaluate_policy(Q_star, transition_matrix, reward_matrix)
        np.save('data_hw2/vf_star.npy', vf_star)

    plot_state_value_fns(vf_clinician, vf_random, vf_nointerv, vf_star)

    pi_clinician = GreedyPolicy(num_states, num_actions, Q_clinician)
    pi_random = GreedyPolicy(num_states, num_actions, Q_random)
    pi_nointerv = GreedyPolicy(num_states, num_actions, Q_nointerv)
    pi_star = GreedyPolicy(num_states, num_actions, Q_star)

    print('plotting deviation')
    plot_deviations_from_clinician(df_cleansed, trajectories, pi_clinician, pi_random, pi_nointerv, pi_star)

    
