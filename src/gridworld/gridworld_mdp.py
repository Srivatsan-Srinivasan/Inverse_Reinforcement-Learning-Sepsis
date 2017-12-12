# general imports
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib
# created by us
import gridworld_project
# The policy outputs the action for each states
from policy.policy import EpsilonGreedyPolicy, GreedyPolicy
from constants_gridworld import IMG_PATH, DATA_PATH
# ---------------------- #
#   Different Domains    #
# ---------------------- #
# You can also create your own!  The interpretation of the different symbols is
# the following:
#
# '#' = wall
# 'o' = origin grid cell
# '.' = empty grid cell
# '*' = goal
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
    '........',
    '........',
    '........',
    '........',
    '........',
    '........',
    '........',
    '.......*']

# ----------------- #
#   Key Functions   #
# ----------------- #
# we need an efficient mdp solver
def policy( state , Q_table , NUM_ACTIONS , epsilon ):
    # an epsilon-greedy policy
    if np.random.random() < epsilon:
        action = np.random.choice( NUM_ACTIONS )
    else:
        action = np.argmax(Q_table[state, :])
        # ties = np.flatnonzero(Q_table[state, :] == Q_table[state, :].max())
        # action = np.random.choice( ties )
    return action

# Update the Q table
def update_Q_Qlearning( Q_table, state , action , reward , new_state , new_action, alpha=0.2, gamma=0.90 ):
    new_action = np.argmax(Q_table[state, :])
    # ties = np.flatnonzero(Q_table[state, :] == Q_table[state, :].max())
    # new_action = np.random.choice( ties )

    Q_table[state, action] = Q_table[state, action] + alpha*(reward + gamma*Q_table[new_state, new_action]- Q_table[state, action])
    return Q_table

def Q_learning(task, NUM_STATES, NUM_ACTIONS, episode_count = 1000000, max_task_iter = np.inf, epsilon = 0.8, verbose = False):
    # Initialize the Q table
    Q_table = np.zeros( ( NUM_STATES , NUM_ACTIONS ) )

    # Initialize transition count table
    transition_count_table = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_STATES))
    iteration = 0

    # Loop until the episode is done
    for episode_iter in range( episode_count ):
        if episode_iter%10 == 0:
            print (episode_iter)
        if iteration >= 5000 and episode_iter >= 5000:
            break
        else:
            # Start the task
            task.reset()
            state = task.observe()
            action = policy( state , Q_table , NUM_ACTIONS , epsilon )
            task_iter = 0

            # Loop until done
            while task_iter < max_task_iter:
                task_iter = task_iter + 1
                # to get a new state
                new_state, reward = task.perform_action( action )
                new_action = policy( new_state , Q_table , NUM_ACTIONS , epsilon )

                if verbose:
                    print (state, action, new_state, new_action)

                # update transition table
                transition_count_table[state, action, new_state] +=1
                Q_table = update_Q_Qlearning(Q_table ,
                                             state , action , reward , new_state , new_action)
                iteration += 1
                # stop if at goal/else update for the next iteration
                if task.is_terminal( state ):
                    break
                else:
                    # Q_table = update_Q_Qlearning(Q_table ,
                    #                          state , action , reward , new_state , new_action)
                    state = new_state
                    action = new_action

    print ("Total iterations:", iteration)
    # convert count to matrix
    transition_matrix = np.zeros(transition_count_table.shape)
    for state in range(NUM_STATES):
        for action in range(NUM_ACTIONS):
            if np.sum(transition_count_table[state, action, :]) > 0:
                transition_matrix[state, action, :] = transition_count_table[state, action, :]/float(np.sum(transition_count_table[state, action, :]))

    # derive optimal policy
    optimal_policy = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_table)
    return optimal_policy, transition_matrix, Q_table

# ---------------- #
#   Run the Task   #
# ---------------- #
# Algorithm Parameters
task_name = happy_grid

action_error_prob = 0.1
pit_reward = -50


task = gridworld_project.GridWorld( task_name,
                    action_error_prob=action_error_prob,
                    rewards={'*': 4, 'moved': 0, 'hit-wall': 0,'X': pit_reward} ,
                    terminal_markers='*' )
NUM_STATES = task.num_states
NUM_ACTIONS = task.num_actions

optimal_policy, transition_matrix, Q_table = Q_learning(task, NUM_STATES, NUM_ACTIONS, verbose = False)

# save results
file_name = DATA_PATH + '3happy_grid_err0_optimal_Q.pickle'
with open(file_name, "wb") as f:
    pickle.dump(Q_table, f, pickle.HIGHEST_PROTOCOL)

file_name = DATA_PATH + '3happy_grid_err0_optimal_policy.pickle'
with open(file_name, "wb") as f:
    pickle.dump(optimal_policy, f, pickle.HIGHEST_PROTOCOL)

file_name = DATA_PATH + '3happy_grid_err0_optimal_transition_matrix.pickle'
with open(file_name, "wb") as f:
    pickle.dump(transition_matrix, f, pickle.HIGHEST_PROTOCOL)

# -------------- #
#   Make Plots   #
# -------------- #
# Note, these are plots that are useful for visualizing the policies
# and the value functions, which can help you identify bugs.  You can
# also use them as a starting point to create the plots that you will
# need for your homework assignment.

# Util to make an arrow
# The directions are [ 'north' , 'south' , 'east' , 'west' ]
# def plot_arrow( location , direction , plot ):

#     arrow = plt.arrow( location[0] , location[1] , dx , dy , fc="k", ec="k", head_width=0.05, head_length=0.1 )
#     plot.add_patch(arrow)

# Useful stats for the plot
row_count = len( task_name )
col_count = len( task_name[0] )
value_function = np.reshape( np.max( Q_table , 1 ) , ( row_count , col_count ) )
policy_function = np.reshape( np.argmax( Q_table , 1 ) , ( row_count , col_count ) )
# wall_info = .5 + np.zeros( ( row_count , col_count ) )
# wall_mask = np.zeros( ( row_count , col_count ) )
# for row in range( row_count ):
#     for col in range( col_count ):
#         if task_name[row][col] == '#':
#             wall_mask[row,col] = 1
# wall_info = np.ma.masked_where( wall_mask==0 , wall_info )

# # Plot the rewards
# plt.subplot( 1 , 2 , 1 )
# plt.plot( episode_reward_set.T )
# plt.title( 'Rewards per Episode (each line is a rep)' )
# plt.xlabel( 'Episode Number' )
# plt.ylabel( 'Sum of Rewards in Episode' )

# value function plot
fig = plt.figure(figsize=(10, 8))
plt.subplot( 1 , 2 , 1 )
plt.imshow( value_function , interpolation='none' , cmap=matplotlib.cm.jet )
plt.colorbar()
# plt.imshow( wall_info , interpolation='none' , cmap=matplotlib.cm.gray )

# policy plot
# plt.imshow( 1 - wall_mask , interpolation='none' , cmap=matplotlib.cm.gray )
for row in range( row_count ):
    for col in range( col_count ):
        # if wall_mask[row][col] == 1:
        #     continue
        if policy_function[row,col] == 0:
            dx = 0; dy = -.5
        if policy_function[row,col] == 1:
            dx = 0; dy = .5
        if policy_function[row,col] == 2:
            dx = .5; dy = 0
        if policy_function[row,col] == 3:
            dx = -.5; dy = 0
        plt.arrow( col , row , dx , dy , shape='full', fc='w' , ec='w' , lw=3, length_includes_head=True, head_width=.2 )

plt.title( 'Expert Policy' , fontsize = 20)
plt.savefig('{}3happygrid_err0_optimal_policy_and_maxQ'.format(IMG_PATH), ppi=300, bbox_inches='tight')
plt.close()
# plt.show( block=False )


