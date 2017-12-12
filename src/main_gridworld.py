# from utils.utils import load_data, extract_trajectories
# from mdp.builder import make_mdp
import numpy as np
# from irl.max_margin import max_margin_learner
# from irl.max_margin_gridworld import max_margin_learner
from gridworld.max_margin_gridworld2 import *
from gridworld.irl_gridworld import *
from gridworld.constants_gridworld import *
import pickle
import gridworld.gridworld_project as env
from gridworld.policy import GreedyPolicy


# grid_world = [   # HERE: Make this one bigger, probably! 
#     '#########',
#     '#..#....#',
#     '#..#..#.#',
#     '#..#..#.#',
#     '#..#.##.#',
#     '#....*#.#',
#     '#######.#',
#     '#o......#',
#     '#########']

happy_grid = [
    'o.......',
    '........',
    '........',
    '........',
    '........',
    '........',
    '........',
    '.......*']


if __name__ == '__main__':

    
    num_iterations = 10
    num_trials = 3
    num_exp_trajectories = 0
    svm_penalty = 300.0
    svm_epsilon = 0.01
    verbose = True
    # experiment_id= 'happygrid_goal1_'
    task_name = happy_grid
    action_error_prob = 0.0
    pit_reward = -50      
    
    task = env.GridWorld( task_name,
                        action_error_prob=action_error_prob, 
                        rewards={'*': 4, 'moved': 0, 'hit-wall': 0,'X': pit_reward} ,
                        terminal_markers='*' )
    NUM_STATES = task.num_states  
    NUM_ACTIONS = task.num_actions
    NUM_FEATURES = 16

    # loading the transition matrix
    file_name = DATA_PATH + '2happy_grid_err0_optimal_transition_matrix.pickle'
    with open(file_name, "rb") as f:
        transition_matrix = pickle.load(f, encoding='latin1')

    # loading pi_expert(optimal policy)
    file_name = DATA_PATH + '2happy_grid_err0_optimal_policy.pickle'
    with open(file_name, "rb") as f:
        pi_expert = pickle.load(f, encoding='latin1')

    # convert the format of pi_expert
    # Q = np.eye(NUM_ACTIONS, dtype=np.float)[pi_expert.astype(int)]
    # Q = Q.reshape(NUM_STATES, NUM_ACTIONS)
    # pi_expert = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q)

    experiment_ids = ['2_2_happygrid_goal4_']
    goal_rewards =  [np.sqrt(NUM_FEATURES)]
    goal_reward_dict = dict(zip(experiment_ids, goal_rewards))

    for experiment_id, goal_reward in goal_reward_dict.items():
        reward_matrix = np.zeros(NUM_STATES)
        for s in range(NUM_STATES):
            if task.is_terminal(s):
                reward_matrix[s] = goal_reward
                # reward_matrix[s] = np.sqrt(NUM_FEATURES)

        sample_initial_state = make_initial_state_sampler_mock(task)
        get_state = make_state_centroid_finder_mock(task, NUM_STATES, NUM_FEATURES)



        # decide which features to include
        # feature_columns = df_centroids.columns

        # check irl/max_margin for implementation
        # check img/ for output
        # max_margin_learner(task, NUM_STATES, NUM_ACTIONS, transition_matrix, reward_matrix, pi_expert, num_iterations=20, epsilon=0.01)
        run_max_margin(task, task_name, transition_matrix, reward_matrix, pi_expert,
                           sample_initial_state, get_state, phi,
                           num_exp_trajectories, svm_penalty, svm_epsilon,
                       num_iterations, num_trials, experiment_id, goal_reward, verbose)