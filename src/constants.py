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

# PCA starts
TRAIN_CLEANSED_PCA_DATA_FILEPATH = DATA_PATH + 'cleansed_data_pca_train.csv'
TRAIN_CENTROIDS_PCA_DATA_FILEPATH = DATA_PATH + 'centroids_data_pca_train.csv'
VALIDATE_CLEANSED_PCA_DATA_FILEPATH = DATA_PATH + 'cleansed_data_pca_val.csv'

TRAIN_TRAJECTORIES_PCA_FILEPATH = DATA_PATH + 'trajectories_pca_train.npy'
TRAIN_TRANSITION_MATRIX_PCA_FILEPATH = DATA_PATH + 'transition_pca_matrix_train.npy'

TRAJECTORIES_PCA_FILEPATH = DATA_PATH + 'trajectories_pca.npy'
TRAJECTORIES_PCA_VASO_FILEPATH = DATA_PATH + 'trajectories_pca_vaso.npy'
TRAJECTORIES_PCA_VASO_SD_LEFT_FILEPATH = DATA_PATH + 'trajectories_pca_vaso_sd_left.npy'
TRAJECTORIES_PCA_VASO_SD_RIGHT_FILEPATH = DATA_PATH + 'trajectories_pca_vaso_sd_right.npy'
TRAJECTORIES_PCA_IV_FILEPATH = DATA_PATH + 'trajectories_pca_iv.npy'
TRAJECTORIES_PCA_IV_SD_LEFT_FILEPATH = DATA_PATH + 'trajectories_pca_iv_sd_left.npy'
TRAJECTORIES_PCA_IV_SD_RIGHT_FILEPATH = DATA_PATH + 'trajectories_pca_iv_sd_right.npy'
TRANSITION_MATRIX_PCA_FILEPATH = DATA_PATH + 'transition_matrix_pca.npy'
# PCA ends

# K prototype starts
TRAIN_CLEANSED_KP_DATA_FILEPATH = DATA_PATH + 'cleansed_data_kp_train.csv'
TRAIN_CENTROIDS_KP_DATA_FILEPATH = DATA_PATH + 'centroids_data_kp_train.csv'
VALIDATE_CLEANSED_KP_DATA_FILEPATH = DATA_PATH + 'cleansed_data_kp_val.csv'
TRAIN_TRAJECTORIES_KP_FILEPATH = DATA_PATH + 'trajectories_kp_train.npy'
TRAIN_TRANSITION_MATRIX_KP_FILEPATH = DATA_PATH + 'transition_kp_matrix_train.npy'
TRAJECTORIES_KP_FILEPATH = DATA_PATH + 'trajectories_kp.npy'
TRAJECTORIES_KP_VASO_FILEPATH = DATA_PATH + 'trajectories_kp_vaso.npy'
TRAJECTORIES_KP_VASO_SD_LEFT_FILEPATH = DATA_PATH + 'trajectories_kp_vaso_sd_left.npy'
TRAJECTORIES_KP_VASO_SD_RIGHT_FILEPATH = DATA_PATH + 'trajectories_kp_vaso_sd_right.npy'
TRAJECTORIES_KP_IV_FILEPATH = DATA_PATH + 'trajectories_kp_iv.npy'
TRAJECTORIES_KP_IV_SD_LEFT_FILEPATH = DATA_PATH + 'trajectories_jp_iv_sd_left.npy'
TRAJECTORIES_KP_IV_SD_RIGHT_FILEPATH = DATA_PATH + 'trajectories_kp_iv_sd_right.npy'
TRANSITION_MATRIX_KP_FILEPATH = DATA_PATH + 'transition_matrix_kp.npy'
# K prototype ends


VALIDATE_FILEPATH = DATA_PATH + 'Sepsis_imp_test.csv'
VALIDATE_CLEANSED_DATA_FILEPATH = DATA_PATH + 'cleansed_data_val.csv'
# we don't use this for now
VALIDATE_CENTROIDS_DATA_FILEPATH = DATA_PATH + 'centroids_data_val.csv'

TRAIN_TRAJECTORIES_FILEPATH = DATA_PATH + 'trajectories_train.npy'
TRAIN_TRANSITION_MATRIX_FILEPATH = DATA_PATH + 'transition_matrix_train.npy'
TRAIN_REWARD_MATRIX_FILEPATH = DATA_PATH + 'reward_matrix_train.npy'

TRAJECTORIES_FILEPATH= DATA_PATH + 'trajectories.npy'
TRAJECTORIES_VASO_FILEPATH = DATA_PATH + 'trajectories_vaso.npy'
TRAJECTORIES_VASO_SD_LEFT_FILEPATH = DATA_PATH + 'trajectories_vaso_sd_left.npy'
TRAJECTORIES_VASO_SD_RIGHT_FILEPATH = DATA_PATH + 'trajectories_vaso_sd_right.npy'
TRAJECTORIES_IV_FILEPATH = DATA_PATH + 'trajectories_iv.npy'
TRAJECTORIES_IV_SD_LEFT_FILEPATH = DATA_PATH + 'trajectories_iv_sd_left.npy'
TRAJECTORIES_IV_SD_RIGHT_FILEPATH = DATA_PATH + 'trajectories_iv_sd_right.npy'
TRANSITION_MATRIX_FILEPATH = DATA_PATH + 'transition_matrix.npy'
REWARD_MATRIX_FILEPATH = DATA_PATH + 'reward_matrix.npy'


STD_BINS_IV_FILEPATH = DATA_PATH + 'std_bins_iv.csv'
STD_BINS_VASO_FILEPATH = DATA_PATH + 'std_bins_vaso.csv'
# since these are experiment-specific, we save them to
# save_path = data/today_date/
PHYSICIAN_Q = 'physician_q'
MDP_OPTIMAL_Q = 'mdp_optimal_q'
IRL_STOCHASTIC_PHYSICIAN_Q_GREEDY = 'irl_stochastic_physician_q_greedy'
IRL_STOCHASTIC_PHYSICIAN_Q_STOCHASTIC ='irl_stochastic_physician_q_stochastic'
IRL_STOCHASTIC_MDP_Q_GREEDY = 'irl_stochastic_mdp_q_greedy'
IRL_STOCHASTIC_MDP_Q_STOCHASTIC = 'irl_stochastic_mdp_q_stochastic'
IRL_GREEDY_PHYSICIAN_Q_GREEDY = 'irl_greedy_physician_q_greedy'
IRL_GREEDY_PHYSICIAN_Q_GREEDY = 'irl_greedy_physician_q_greedy'
IRL_GREEDY_PHYSICIAN_Q_STOCHASTIC ='irl_greedy_physician_q_stochastic'
IRL_GREEDY_MDP_Q_GREEDY = 'irl_greedy_mdp_q_greedy'
IRL_GREEDY_MDP_Q_STOCHASTIC = 'irl_greedy_mdp_q_stochastic'

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
ETC = ['age', 'charttime']
COLS_TO_BE_NORMALIZED = EXT_MEASUREMENTS + BLOOD_SAMPLES + ['age', 'cumulated_balance_tev'] + ['Shock_Index', 'GCS']
COLS_TO_BE_NORMALIZED_PLUS = EXT_MEASUREMENTS + BLOOD_SAMPLES + PSEUDO_OBSERVATIONS + SUMMARY_INDEX + ['age', 'gender', 're_admission', 'cumulated_balance_tev']
# NEW UPDATE TO MAKE IT COMPATIBLE WITH PHI(S)
#BINARY_COLS = ['gender', 're_admission', 'sedation', 'mechvent', 'rrt', 'died_in_hosp', 'mortality_90d']
#COLS_TO_BE_NORMALIZED = list(set(COLS_TO_BE_NORMALIZED) - set(BINARY_COLS))
COLS_TO_BE_LOGGED=  ['SpO2','Glucose','BUN','Creatinine', 'SGOT', 'SGPT','Total_bili','WBC_count',
        'Platelets_count', 'PTT','PT','INR','paO2','paCO2','Arterial_lactate','PaO2_FiO2',
        'GCS', 'Shock_Index']

ALREADY_NORMAL_LOGGED = list(set(COLS_TO_BE_NORMALIZED) - set(COLS_TO_BE_LOGGED))

INTEGER_COLS = OUTCOMES + ['age', 'bloc', 'SOFA', 'state', 'icustayid']

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
