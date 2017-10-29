# -*- coding: utf-8 -*-
"""
@author: Camilo

CS282r HW3
Problem 3: Designing a similarity measure between patients and finding what action to take
with a specific patient on a specific state by looking at what was done to surviving similar patients.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import functions_hw1 as fhw1
import warnings
import random
from timeit import default_timer as timer
from tqdm import tqdm
import utils
import pickle


############### FUNCTIONS FOR PROBLEM 3 ##################

def get_minimum_timeblocs(p1, p2):
    """ Get min timeblocs between patient 1 and patient 2"""
    
    return min(p1['icustayid'].count(), p2['icustayid'].count())


def get_patients_that_visited_state(patient_df, ids, s):
    """ 
    Returns action and id of patients in ids that visited s. 
    
    INPUTS
    ======
    patient_df: dataframe of patients. must have a 'states' field.
    ids: vector of all patient icustayids that we want to check.
    s: state that we need to check
    """
    patient_required_df = patient_df[patient_df['icustayid'].isin(ids)]
    
    state_rows = patient_required_df[patient_required_df['state']==s]
    
    if state_rows.empty:
        print('WARNING(get_patients_that_visited_state)')
        warnings.warn('\nWARNING(get_patients_that_visited_state): STATE_ROWS EMPTY. NO STATES CORRESPOND WITH S= '+str(s)+' IN PATIENTS PROVIDED\n', UserWarning)
        return None
    else:    
        ids_patients_visited_s = state_rows['icustayid'].unique()

    return np.array(ids_patients_visited_s)
        
def get_surviving_patients(patient_df, ids):
    """
    Returns ids of surviving patients in list 'ids' as a numpy array.
    patient_df should have a field 'mortality_90d'
    """
    
    patient_required_df = patient_df[patient_df['icustayid'].isin(ids)]
    
    surv_rows = patient_required_df[patient_required_df['mortality_90d']==0]    
    
    if surv_rows.empty:
        print('WARNING(get_surviving_patients)')
        warnings.warn('\nWARNING(get_surviving_patients): NO SURVIVING PATIENTS IN IDS PROVIDED\n', UserWarning)
        return None
    else:    
        ids_surv = surv_rows['icustayid'].unique()
    
    return np.array(ids_surv)

def get_k_most_similar_patients(similarity_dict, icustayid, ids, k):
    """
    Returns ids of k most similar patients in list 'ids' to patient
    with id icustayid.
    """
    
    similarity_scores = []
    icustayid = int(icustayid)
    for i in [int(j) for j in ids]:
        if i > icustayid:
            similarity_scores.append((similarity_dict[icustayid][i],i))
        elif i < icustayid:
            similarity_scores.append((similarity_dict[i][icustayid],i))
        else:
            similarity_scores.append((0,i))
    
        
#    full_ids = sepsis_df['icustayid'].unique()
#    
#    id_idx = [int(np.argwhere(full_ids==j)) for j in ids]
#    
##    print('id_idx:',id_idx)
#    
#    similarity_scores = [ similarity_matrix[int(np.argwhere(full_ids==icustayid)),i] for i in id_idx ]
       
    sorted_scores = sorted(similarity_scores, key=lambda x: x[0])
    
#    print('Sorted_scores',sorted_scores)
    
    most_sim_ids = [x[1] for x in sorted_scores]
    
#    print('Most sim ids:',most_sim_ids)
    
    if k > len(most_sim_ids):
        warnings.warn("\nWARNING(get_k_most_similar_patients): NOT ENOUGH SIMILAR PATIENTS. RETURNING ONLY "+str(len(most_sim_ids))+"\n", UserWarning)
        return most_sim_ids
    return most_sim_ids[:k]
        
def get_actions_for_state(patient_df,ids,s):
    """
    Returns np.array of actions taken in state s by patients in list ids.
    """
    
    patient_required_df = patient_df[patient_df['icustayid'].isin(ids)]
    
    state_rows = patient_required_df[patient_required_df['state']==s]
    
    if state_rows.empty:
        warnings.warn('\nWARNING(get_actions_for_state): STATE_ROWS EMPTY. NO STATES CORRESPOND WITH S= '+str(s)+' IN PATIENTS PROVIDED\n', UserWarning)
        return None
    
    actions = np.array(state_rows['action_bin'])
#    print('Actions taken by similar patients on state s:',actions)
    
    return actions
    
    
def compute_recommended_action(actions,mean=0):
    """
    Computes recomended action from array by choosing either the mode,
    or the closest value for the mean
    """
    
    if mean:
        idx = (np.abs(actions-np.mean(actions))).argmin()
        return actions[idx]
    else:
        # Mode is useful because if there is no more likely value, it returns the smallest one, which makes sense with the no intervention idea
        m= stats.mode(actions)
        return int(m[0])
    

def compute_recom_action_based_on_similarity(row,index,ids,similarity_matrix):
    """
    Main function for computing recomended actions based on a measure of similarity.
    Similarity matrix needs to be computed before.
    
    Main idea: for each timebloc of each patient, this function looks at the patients that
    visited the state corresponding to this timebloc, and then selects the patients that
    survived. We then get the top 10 most similar ones to our original patient, 
    and recomend the mean/mode of actions taken for that set of 100 similar patients.
    """
#    if not index%100:
#        print('\n\nStarting recom action computation for row:',index,'. STATE = ',row['state'],'ID =',row['icustayid'],'ACTION =',row['action_bin'])
#    print('getting ids for patient that visited state')
    ids_visited_state = get_patients_that_visited_state(sepsis_df, ids, row['state'])
#    print('------- IDS THAT VISITED S:',ids_visited_state)
    if(ids_visited_state is None):
        recom_action_for_this_timebloc = random.randint(0,24)
#        print('Recommended random action for row',index,'is:',recom_action_for_this_timebloc)
        return recom_action_for_this_timebloc
        
        
#    print('getting survivors')
    ids_surv = get_surviving_patients(sepsis_df, ids_visited_state)
#    print('------- IDS THAT VISITED S AND SURV:',ids_surv)
    if(ids_surv is None):
        recom_action_for_this_timebloc = random.randint(0,24)
#        print('Recommended random action for row',index,'is:',recom_action_for_this_timebloc)
        return recom_action_for_this_timebloc
        
        
#    print('getting 100 most similar')
    ids_most_similar = get_k_most_similar_patients(similarity_matrix, row['icustayid'], ids_surv, 100)
#    print('------- IDS THAT VISITED S AND SURV AND ARE SIMILAR:',ids_most_similar)
#    print('getting list of similar actions')
    actions_most_similar = get_actions_for_state(sepsis_df, ids_most_similar, row['state'])
#    print('getting recomended action')
    recom_action_for_this_timebloc = compute_recommended_action(actions_most_similar)
#    print('Recommended action for row',index,'is:',recom_action_for_this_timebloc)
   
    return recom_action_for_this_timebloc

   
def compute_similarity_distance_dict(scaled_df,ids):
    start = timer()
    # create a time series containing matrices
    timeseries = scaled_df.groupby('icustayid').apply(lambda x:x.values)
    
    # Create gamma vector with decreasing weights per timestep
    timesteps = [m.shape[0] for m in timeseries.values]
    gamma = [0.95**t for t in range(max(timesteps)+1)]
    
    # in the end this dict will be a mapping {patient_id: [(similar_patient, distance) for similar_patient in the_top_200_nearest_patients]}
    dists = {id: {} for id in ids}
    # iterate the series with iteritems which is superfast
    for i1, (patient1, values1) in tqdm(enumerate(timeseries.iteritems())):
        # patient1 is the id of the patient (icustayid) and values1 is the matrix containing the timeseries
        for i2, (patient2, values2) in enumerate(timeseries.iloc[i1+1:].iteritems()):
            # compare the time series on their n first time steps. It assumes that the time steps are the same
            n = min(values1.shape[0], values2.shape[0])
            dists[patient1][patient2] = np.dot(gamma[:n],np.sum((values1[:n, 1:] - values2[:n, 1:])**2,1))/n  # note that there is a -1 because the last column is the icustayid
            #print(dists[patient1][patient2])
            
        # take only the first 200 nearest
        #dists[patient1] = sorted(list(dists[patient1].items()), key=lambda x: x[1])[:200]  
    end = timer()
    print('Time to compute sim_dict for',len(timeseries),'patients was:',end-start)
    return dists
            
def compute_similarity_matrix(scaled_df,ids):
    """
    Given scaled data of N patients, and N ids, compute the similarity matrix
    by calculating, for each (p1,p2) pair, the weighted sum of the difference between
    each feature, summed for each time block with decay coefficient gamma.
    """

    # Arbitrarily chosen gamma to give more weight to the first time blocs
    gamma = 0.9
    weights = np.ones(len(scaled_df.loc[:,'age':].count()))
    
    # There are 40 features that we want to use for our similarity matrix
    assert(len(weights) == 40)
    
    # Initializing ids x ids sim matrix
    patient_sim_matrix = np.zeros((len(ids),len(ids)))
    
    oldtime = timer()
    for idx_p1, p1_id in enumerate(ids):
        if not idx_p1%10:   # Print progress
            time = timer()
            print('Computing sim_matrix row for patient (idx_p1):',idx_p1)
            print('Time for this iteration:',time-oldtime)
            oldtime = time
                    
        for idx_p2 in range(idx_p1):
            
            # Getting patients from scaled_df DF. These patient dataframes still have the icustayid column
            p1 = scaled_df[scaled_df['icustayid'] == ids[idx_p1]]#p1_id]
            p2 = scaled_df[scaled_df['icustayid'] == ids[idx_p2]]
            min_t  = get_minimum_timeblocs(p1, p2)
            p1=p1.reset_index(drop=True)
            p2=p2.reset_index(drop=True)
            
            # Calculating differences with .as_matrix. This is probably way too time consuming.
            differences = pd.DataFrame(p1.loc[0:min_t-1,'age':].as_matrix() - p2.loc[0:min_t-1,'age':].as_matrix(), columns=p1.loc[0:min_t-1,'age':].columns)
                       
            # Computing similarity matrix
            for t in range(min_t):  
                patient_sim_matrix[idx_p1][idx_p2] += gamma**t * np.dot(weights, np.square(differences.loc[t,:].as_matrix()))
                patient_sim_matrix[idx_p1][idx_p2] = patient_sim_matrix[idx_p1][idx_p2]/min_t
            # Adding symmetric values
            patient_sim_matrix[idx_p2][idx_p1] = patient_sim_matrix[idx_p1][idx_p2]
                
    
    print('FINISHED. SIMILARITY MATRIX IS:',patient_sim_matrix[:10][:10])
    
    return patient_sim_matrix

def save_similarity_dict(sim_dict):
    print('Saving similarity dict...')
    np.save('data/sim_distance_dict.npy', sim_dict) 
    print('Done')

def scale_data(numerical_data):
    scaled_df = (numerical_data.loc[:,'age':] - numerical_data.loc[:,'age':].mean())/numerical_data.loc[:,'age':].std()
    scaled_df.insert(0, 'icustayid', numerical_data['icustayid'])
    #assert(len(scaled_df)==252204) # 252204 timeblocs
    return scaled_df

def create_policy_table(actions, states, num_actions, num_states):
    
    p_table = np.zeros((num_states, num_actions))
    
    for s in range(num_states):
        action = stats.mode(actions[states == s])
        
        for a in range(num_actions):
            if a == action[0]:
                p_table[s,a] = 1
    
    return p_table
    
def plot_V(Vs,labels):
    
    plt.figure()
    
    for i in range(len(Vs)):
        V_sorted = np.sort(Vs[i])
        plt.plot(V_sorted, label = labels[i])
    
    plt.title('Value functions vs states')
    plt.xlabel('states')
    plt.ylabel('V(s)')
    plt.legend()
    
    plt.savefig( 'Report/V_plot_problem_3.png' )

def compute_dif_IV_VP(IV_bin_values, VP_bin_values, actions, recomended_actions, add_to_df = False):
    
#    actions_sequence = median_dose_vaso__sequence__discretized * bins_num + input_4hourly__sequence__discretized
#    action = VP * bins_num + IV

    IV_bin_phys = [i%5 for i in actions]
    VP_bin_phys = [int((i-IV_bin_phys[k])/5) for k,i in enumerate(actions)]
    
    IV_bin_recom = [i%5 for i in recomended_actions]
    VP_bin_recom = [int((i-IV_bin_recom[k])/5) for k,i in enumerate(recomended_actions)]
    
#    print(VP_bin, sepsis_df['action_bin'])
    
    assert(max(VP_bin_recom)<6)
    assert(max(IV_bin_recom)<6)
    
    IV_physician = np.array([IV_bin_values[i] for i in IV_bin_phys])
    IV_recom = np.array([IV_bin_values[i] for i in IV_bin_recom])
    
    VP_physician = np.array([VP_bin_values[i] for i in VP_bin_phys])
    VP_recom = np.array([VP_bin_values[i] for i in VP_bin_recom])
    
    print('IV PHYSICIAN',IV_physician[:10])
    print('IV RECOM',IV_recom[:10])
    
    IV_dif = IV_physician - IV_recom
    VP_dif = VP_physician - VP_recom
    
    if add_to_df:
        sepsis_df['IV_dif'] = IV_dif
        sepsis_df['VP_dif'] = VP_dif
    else:
        return IV_dif, VP_dif, IV_bin_phys, IV_bin_recom, VP_bin_phys, VP_bin_recom, IV_recom, VP_recom
    
    
def compute_mortality_dif_for_each_action_dif(df):
     
#    IV_dif = sepsis_df['IV_dif']
    
    mortality_dif_IV_df = df.groupby(['IV_dif']).mean()
    mortality_dif_VP_df = df.groupby(['VP_dif']).mean()
#    
#    print(mortality_dif_IV_df.head(50))
#    print(mortality_dif_IV_df.index)
    return (mortality_dif_IV_df.index.tolist(),mortality_dif_IV_df['mortality_90d']), (mortality_dif_VP_df.index.tolist(),mortality_dif_VP_df['mortality_90d'])

def transform_Q_to_action_row(Q_table, sepsis_df, label):
    
    df = pd.DataFrame()
    df[label] = np.argmax(Q_table[sepsis_df['state'],:],1)
    
    print('len(df)',len(df))
    
    return df

def plot_mortality_rate_vs_dif(diff, mortality_dif, title, xlabel, ylabel):
    plt.figure()
    plt.plot(diff, mortality_dif)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

############### SCRIPT FOR PROBLEM 3 ##################

load =1
save = 0

sepsis_df = pd.read_csv('data/df_cam.csv')

# Separating only numerical data
numerical_data = sepsis_df.drop(['charttime','gender','elixhauser','re_admission', 'SIRS'],1)
numerical_data = numerical_data.loc[:,:'PaO2_FiO2']

# Scaling numerical data
scaled_df = scale_data(numerical_data)

# Saving to csv

if save:
    scaled_df.to_csv('data/Sepsis_imp_scaled.csv')

# Generating id vector
ids = sepsis_df['icustayid'].unique()
print('Amount of different patients:', len(ids))
#assert(len(ids) == 19275) # 19275 different patients

# Load Dictionary of distances

if load:
    with open('data/sim_distance_dict', 'rb') as f:
        sim_distance_df = pickle.load(f)

else:
    sim_distance_dict = compute_similarity_distance_dict(scaled_df,ids)
    save_similarity_dict(sim_distance_dict)        

if load:
    recom_actions = np.load('data/recom_actions.npy')
    sepsis_df['recom_action'] = recom_actions
else:
    recom_actions = []
    for index, row in tqdm(sepsis_df.iterrows()):
        recom_actions.append( compute_recom_action_based_on_similarity(row,index,ids,sim_distance_dict) )
     
sepsis_df['recom_action'] = recom_actions
if save:
    sepsis_df.to_csv('data/Sepsis_imp_with_actions.csv')   

print('RECOM ACTION COMPUTATION FINISHED.')


GAMMA = .95
T = np.load('data/transition_matrix.npy')
R = np.load('data/reward_table.npy')
V_star = np.load('data/vf_star.npy')
V_nointerv = np.load('data/vf_nointerv.npy')
V_random = np.load('data/vf_random.npy')
V_clin = np.load('data/vf_clinician.npy')

STATE_COUNT = T.shape[0]
ACTION_COUNT = T.shape[1]

#print('ACTION COUNT:', ACTION_COUNT)
#print('STATE COUNT:', STATE_COUNT)
assert(ACTION_COUNT == 25)
assert(STATE_COUNT == 753)




# With the recomended actions, we can now get our policy table:   
policy_table_recom_actions = create_policy_table(sepsis_df['recom_action'],sepsis_df['state'], ACTION_COUNT, STATE_COUNT)

# And we solve the MDP with our new policy, using T and R from Problem 2.
Q_recom, V_recom = fhw1.solve_MDP_equations(T, R, GAMMA, policy_table_recom_actions, STATE_COUNT, ACTION_COUNT)

# With the calculated V, we can plot the performance over states compared to other policies
plot_V([V_recom, V_star, V_nointerv, V_random, V_clin], ['V_recom', 'V_star', 'V_nointervention', 'V_random', 'V_clinician'])

# We now search for the bins in IV and VP for our 25 actions
ac, IV_bin_values, VP_bin_values = utils.discretize_actions(sepsis_df['input_4hourly_tev'], sepsis_df['median_dose_vaso'])

# We compute the numerical difference of VP and IV dosage
IV_dif, VP_dif, IV_bin_phys, IV_bin_recom, VP_bin_phys, VP_bin_recom, IV_recom, VP_recom = compute_dif_IV_VP(IV_bin_values, VP_bin_values, sepsis_df['action_bin'], sepsis_df['recom_action'])

   
# Add those values to the dataframe
sepsis_df['IV_dif'] = IV_dif
sepsis_df['VP_dif'] = VP_dif
sepsis_df['IV_bin_phys'] = IV_bin_phys
sepsis_df['IV_bin_recom'] =  IV_bin_recom
sepsis_df['VP_bin_phys'] = VP_bin_phys
sepsis_df['VP_bin_recom'] = VP_bin_recom
sepsis_df['IV_recom'] = IV_recom
sepsis_df['VP_recom'] = VP_recom

#sepsis_df.to_csv('data/Sepsis_imp_with_actions_and_dosages.csv')   

# We return the mortality for each of the 5 differences
(diff_IV, mortality_dif_IV),(diff_VP, mortality_dif_VP) = compute_mortality_dif_for_each_action_dif(sepsis_df)

# IV difference vs mortality plot
plot_mortality_rate_vs_dif(diff_IV, mortality_dif_IV,
                           'Mortality rate vs differences between recommended IV dosage and physician dosage',
                           'IV dosage difference',
                           'Mortality rate')

# VP difference vs mortality plot
plot_mortality_rate_vs_dif(diff_VP, mortality_dif_VP,
                           'Mortality rate vs differences between recommended VP dosage and physician dosage',
                           'VP dosage difference',
                           'Mortality rate')


###### PLOTS FOR PROBLEM 2 ######

Q_star = np.load('data/q_star.npy')
Q_nointerv = np.load('data/q_nointerv.npy')
Q_random = np.load('data/q_random.npy')
Q_clin = np.load('data/q_clinician.npy')

Q_labels = ['star_action','nointerv_action','random_action', 'clin_action']

for i,Q in enumerate( [Q_star, Q_nointerv, Q_random, Q_clin]):
    df = transform_Q_to_action_row(Q, sepsis_df, Q_labels[i])
    df['mortality_90d'] = sepsis_df['mortality_90d'].copy()
    IV_dif, VP_dif, IV_bin_phys, IV_bin_recom, VP_bin_phys, VP_bin_recom, IV_recom, VP_recom = compute_dif_IV_VP(IV_bin_values, VP_bin_values, sepsis_df['action_bin'], df[Q_labels[i]])
    df['IV_dif'] = IV_dif
    df['VP_dif'] = VP_dif
    (diff_IV, mortality_dif_IV),(diff_VP, mortality_dif_VP) = compute_mortality_dif_for_each_action_dif(df)
    plot_mortality_rate_vs_dif(diff_IV, mortality_dif_IV,
                           'Mortality rate vs differences between '+Q_labels[i]+' IV dosage and physician dosage',
                           'IV dosage difference',
                           'Mortality rate')
    plt.savefig( 'Report/Mortality_rate_vs_IV_diff_'+Q_labels[i]+'.png' )
    plot_mortality_rate_vs_dif(diff_VP, mortality_dif_VP,
                           'Mortality rate vs differences between '+Q_labels[i]+' VP dosage and physician dosage',
                           'VP dosage difference',
                           'Mortality rate')
    plt.savefig( 'Report/Mortality_rate_vs_VP_diff_'+Q_labels[i]+'.png' )