from utils.utils import load_data, extract_trajectories
from mdp.builder import make_mdp
# from irl.max_margin import max_margin_learner
from irl.max_margin_gridworld import max_margin_learner
from constants import *


if __name__ == '__main__':
    # loading the whole data
    df, df_cleansed, df_centroids = load_data()
    trajectories = extract_trajectories(df_cleansed, NUM_PURE_STATES)
    transition_matrix, reward_matrix = make_mdp(trajectories, NUM_STATES, NUM_ACTIONS)
    
    # decide which features to include
    feature_columns = df_centroids.columns

    # check irl/max_margin for implementation
    # check img/ for output
    max_margin_learner(df_cleansed, df_centroids, feature_columns,
                           trajectories, transition_matrix, reward_matrix,
                           num_iterations=20, epsilon=0.01)
