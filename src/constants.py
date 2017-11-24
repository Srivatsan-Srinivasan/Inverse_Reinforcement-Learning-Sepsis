import seaborn as sns
import numpy as np

NUM_ACTIONS = 25
# there will be 752 states in total
# plus 2 bc. terminal states(survive vs. died in hosp)
NUM_PURE_STATES = 750
NUM_TERMINAL_STATES = 2
NUM_STATES = NUM_PURE_STATES + NUM_TERMINAL_STATES
# terminal states index
TERMINAL_STATE_ALIVE = NUM_STATES - 2
TERMINAL_STATE_DEAD = NUM_STATES - 1

IMG_PATH = 'img/'
DATA_PATH = 'data/'
FILEPATH = DATA_PATH + 'sepsis.csv'

TRAIN_FILEPATH = DATA_PATH + 'Sepsis_imp_train.csv'
TRAIN_CLEANSED_DATA_FILEPATH = DATA_PATH + 'cleansed_data_train.csv'
TRAIN_CENTROIDS_DATA_FILEPATH = DATA_PATH + 'centroids_data_train.csv'

VALIDATE_FILEPATH = DATA_PATH + 'Sepsis_imp_test.csv'
VALIDATE_CLEANSED_DATA_FILEPATH = DATA_PATH + 'cleansed_data_val.csv'
# we don't use this for now
VALIDATE_CENTROIDS_DATA_FILEPATH = DATA_PATH + 'centroids_data_val.csv'

TRAIN_TRAJECTORIES_FILEPATH = DATA_PATH + 'trajectories_train.npy'
TRAIN_TRANSITION_MATRIX_FILEPATH = DATA_PATH + 'transition_matrix_train.npy'
TRAIN_REWARD_MATRIX_FILEPATH = DATA_PATH + 'reward_table_train.npy'

VALIDATE_TRAJECTORIES_FILEPATH = DATA_PATH + 'trajectories_val.npy'
VALIDATE_TRANSITION_MATRIX_FILEPATH = DATA_PATH + 'transition_matrix_val.npy'
VALIDATE_REWARD_MATRIX_FILEPATH = DATA_PATH + 'reward_table_val.npy'

PHYSICIAN_Q = DATA_PATH + 'physician_q.npy'
MDP_OPTIMAL_Q = DATA_PATH + 'mdp_optimal_q.npy'
IRL_PHYSICIAN_Q = DATA_PATH + 'irl_physician_q.npy'
IRL_MDP_Q = DATA_PATH + 'irl_mdp_q.npy'

Q_STAR_FILEPATH = DATA_PATH + 'q_star.npy'
Q_CLINICIAN_FILEPATH = DATA_PATH + 'q_clinician.npy'
VF_CLINICIAN_FILEPATH  = DATA_PATH + 'vf_clinician.npy'
VF_RANDOM_FILEPATH = DATA_PATH + 'vf_random.npy'
CRITICAL_VALUES = ['HR', 'RR', 'SysBP']
CRITICAL_VALUES_LARGER = ['age', 'HR', 'MeanBP', 'Temp_C', 'FiO2_1']

ALL_VALUES = ['bloc', 'icustayid', 'charttime', 'gender', 'age', 'elixhauser', 're_admission',
        'SOFA', 'SIRS', 'Weight_kg', 'GCS', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'Shock_Index', 'RR',
        'SpO2', 'Temp_C', 'FiO2_1', 'Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN',
        'Creatinine', 'Magnesium', 'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT',
        'Total_bili', 'Albumin', 'Hb', 'WBC_count', 'Platelets_count', 'PTT', 'PT', 'INR',
        'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2',
        'median_dose_vaso', 'max_dose_vaso', 'input_total_tev', 'input_4hourly_tev', 'output_total', 'output_4hourly', 'cumulated_balance_tev', 'sedation', 'mechvent', 'rrt', 'died_in_hosp', 'mortality_90d']

BLOOD_SAMPLES = ['Potassium', 'Sodium', 'Chloride', 'Glucose', 'BUN', 'Creatinine', 'Magnesium',
        'Calcium', 'Ionised_Ca', 'CO2_mEqL', 'SGOT', 'SGPT', 'Total_bili', 'Albumin', 'Hb',
        'WBC_count', 'Platelets_count', 'PTT', 'PT', 'INR', 'Arterial_pH', 'paO2', 'paCO2',
        'Arterial_BE', 'Arterial_lactate', 'HCO3', 'PaO2_FiO2']

EXT_MEASUREMENTS = ['Weight_kg', 'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'SpO2',
        'Temp_C', 'FiO2_1','output_total', 'output_4hourly']

PSEUDO_OBSERVATIONS = ['sedation', 'mechvent', 'rrt']
INTERVENTIONS = ['input_total_tev','input_4hourly_tev','median_dose_vaso' ,'max_dose_vaso']

TRUE_INTERVENTIONS = ['median_dose_vaso', 'max_dose_vaso', 'input_total_tev', 'input_4hourly_tev', 'cumulated_balance_tev', 'sedation', 'mechvent', 'rrt']

# shock_index is a ratio 
SUMMARY_INDEX = ['elixhauser', 'SOFA', 'SIRS', 'GCS', 'Shock_Index']

TIME = ['bloc', 'charttime']

OUTCOMES = ['died_in_hosp', 'mortality_90d']

# this is a bit hand-wavy as we're hardcoding this
CATEGORICAL_ORDINAL = SUMMARY_INDEX + ['bloc']
CATEGORICAL_NOMINAL = ['icustayid', 'gender', 're_admission'] + OUTCOMES
# numerical should not be scale-invariant?
# does it make sense to interprete the mean of these multiplied by some constant? 
# if yes, it should go here
# we don't normalize interventions
#COLS_TO_BE_NORMALIZED = EXT_MEASUREMENTS + BLOOD_SAMPLES + PSEUDO_OBSERVATIONS + SUMMARY_INDEX 
# TODO: this does not make sense but will fix later
# when we do clustering other than kmeans
COLS_TO_BE_NORMALIZED = EXT_MEASUREMENTS + BLOOD_SAMPLES + PSEUDO_OBSERVATIONS + SUMMARY_INDEX + ['age', 'gender', 're_admission', 'cumulated_balance_tev']
ETC = ['age', 'charttime']

COLS_TO_BE_LOGGED=  ['SpO2','Glucose','BUN','Creatinine', 'SGOT', 'SGPT','Total_bili','WBC_count',
        'Platelets_count', 'PTT','PT','INR','paO2','paCO2','Arterial_lactate','PaO2_FiO2',
        'GCS', 'Shock_Index']

ALREADY_NORMAL_LOGGED = list(set(COLS_TO_BE_NORMALIZED) - set(COLS_TO_BE_LOGGED))

INTEGER_COLS = OUTCOMES + ['age', 'bloc', 'SOFA', 'state', 'icustayid', 'bloc']

# gender, age, readmission, rrt, vent, sedation could be excluded
COLS_NOT_FOR_CLUSTERING = ['icustayid', 'charttime', 'bloc']
BINARY_COLS = ['gender', 're_admission', 'rrt', 'mechvent', 'sedation']

# plotting
palette = sns.color_palette("muted", 70)

LINESTYLES = ['-', '--', ':']
COLORS = palette

left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .5     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

# MDP
TERMINAL_MARKER = -1
