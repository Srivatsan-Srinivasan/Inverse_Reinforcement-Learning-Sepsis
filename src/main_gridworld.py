from utils.utils import load_data, extract_trajectories
from mdp.builder import make_mdp
import numpy as np
# from irl.max_margin import max_margin_learner
from irl.max_margin_gridworld import max_margin_learner
# from constants import *
import pickle
import gridworld_project
from policy.policy import GreedyPolicy


grid_world = [   # HERE: Make this one bigger, probably! 
    '#########',
    '#..#....#',
    '#..#..#.#',
    '#..#..#.#',
    '#..#.##.#',
    '#....*#.#',
    '#######.#',
    '#o......#',
    '#########']

if __name__ == '__main__':


    task_name = grid_world 
    action_error_prob = 0.1
    pit_reward = -50      
    
    task = gridworld_project.GridWorld( task_name,
                        action_error_prob=action_error_prob, 
                        rewards={'*': 50, 'moved': -1, 'hit-wall': -1,'X': pit_reward} ,
                        terminal_markers='*' )
    NUM_STATES = task.num_states  
    NUM_ACTIONS = task.num_actions

    # loading the transition matrix
    file_name = '/Users/linyingzhang/Desktop/grid_world_optimal_transition_matrix.pickle'
    with open(file_name, "rb") as f:
        transition_matrix = pickle.load(f, encoding='latin1')

    # loading pi_expert(optimal policy)
    file_name = '/Users/linyingzhang/Desktop/grid_world_optimal_policy.pickle'
    with open(file_name, "rb") as f:
        pi_expert = pickle.load(f, encoding='latin1')

        # loading pi_expert(optimal policy)
    file_name = '/Users/linyingzhang/Desktop/grid_world_optimal_Q.pickle'
    with open(file_name, "rb") as f:
        Q_table = pickle.load(f, encoding='latin1')

    Q = np.eye(NUM_ACTIONS, dtype=np.float)[pi_expert.astype(int)]
    Q = Q.reshape(NUM_STATES, NUM_ACTIONS)

    pi_expert = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q)

    reward_matrix = np.zeros(NUM_STATES)
    
    # decide which features to include
    # feature_columns = df_centroids.columns

    # check irl/max_margin for implementation
    # check img/ for output
    max_margin_learner(task, NUM_STATES, NUM_ACTIONS, transition_matrix, reward_matrix, pi_expert,
                   num_iterations=50, epsilon=0.01)