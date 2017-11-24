import numpy as np
import numba as nb
import logging
import itertools
from policy.policy import EpsilonGreedyPolicy, GreedyPolicy
from learners.monte_carlo_on_policy import run_mc_actor
from constants import TERMINAL_STATE_ALIVE, TERMINAL_STATE_DEAD

# Update the Q table 
def update_Q_Qlearning( Q_table, state , action , reward , new_state , new_action, alpha=0.5, gamma=0.95 ):
    new_action = np.random.choice(np.flatnonzero(Q_table[ new_state , : ] == Q_table[ new_state , : ].max()))
    Q_table[state, action] = Q_table[state, action] + alpha*(reward + gamma*Q_table[new_state, new_action]- Q_table[state, action])
    # FILL THIS IN 
    return Q_table 

def Q_learning_solver_for_irl(task, transition_matrix, reward_matrix, NUM_STATES, NUM_ACTIONS, episode_count = 500, max_task_iter = np.inf, epsilon = 0.2):
    # Initialize the Q table 
    Q_table = np.zeros( ( NUM_STATES , NUM_ACTIONS ) )
    iteration = 0

    # Loop until the episode is done 
    for episode_iter in range( episode_count ):
        print ("episode_iter", episode_iter)
        if iteration >= 3000 and episode_iter >= 100:
            break
        else:
            # Start the task 
            task.reset()
            state = task.observe() 
            print ("initial state", state)
            action = policy( state , Q_table , NUM_ACTIONS , epsilon ) 
            task_iter = 0 

            # Loop until done
            while task_iter < max_task_iter:
                task_iter += 1

                # to get a new state
                # new_state, reward = task.perform_action( action )
                reward = reward_matrix[state]
                t_probs = np.copy(transition_matrix[state, action, :])
                new_state = np.random.choice(NUM_STATES, p=t_probs)
                new_action = policy( new_state , Q_table , NUM_ACTIONS , epsilon ) 
                if iteration%5000 == 0:
                    print ("iteration", iteration )
                    print ("s,a,s,a", state, action, new_state, new_action)

                # update Q_table    
                Q_table = update_Q_Qlearning(Q_table , 
                                             state , action , reward , new_state , new_action)
                # stop if at goal/else update for the next iteration 
                if task.is_terminal( state ):
                    break
                else:
                    state = new_state
                    action = new_action
                                # store the data
                iteration += 1

    # derive optimal policy
    optimal_policy = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_table)
    return optimal_policy, Q_table


def evaluate_policy_monte_carlo(env, pi, gamma=0.99, num_episodes=100):
    rewards = []
    for _ in range(num_episodes):
        exps = run_mc_actor(env, pi, max_local_iter=500)
        G = np.sum([(gamma**t)*e[2] for t, e in enumerate(exps)])
        rewards.append(G)
    return np.mean(rewards)

@nb.jit
def iterate_value(Q_table, transition_matrix, reward_table, gamma=0.95, theta=0.1, max_iter=100):
    # TODO: fix this. does not work now
    num_states = Q.shape[0]
    v = np.zeros(num_states)
    for n in range(max_iter):
        v_temp = v
        # s a  a
        Q = transition_matrix.dot(np.expand_dims(reward_table, axis=1) + gamma * Q)
        if np.abs(v_temp - v).max() < theta:
            print('policy evaluated after {} steps'.format(n))
            break
    return v


@nb.jit
def evaluate_policy(Q, transition_matrix, reward_table, gamma=0.99, theta=1e-1, max_iter=300):
    num_states = Q.shape[0]
    v = np.zeros(num_states)
    for n in range(max_iter):
        v_temp = v
        v = np.zeros(num_states)
        for s in range(num_states):
            # evaluate this policy's action choices
            ties = np.flatnonzero(Q[s, :] == Q[s, :].max())
            a = np.random.choice(ties)
            r = reward_table[s]
            v[s] = r + gamma * transition_matrix[s, a, :].dot(v_temp)
        max_delta = np.linalg.norm(v_temp - v, np.inf)
        if max_delta < theta:
            print('policy evaluated after {} steps'.format(n))
            break
    return v

@nb.jit
def iterate_policy(Q, transition_matrix, reward_matrix, gamma=0.99, theta=1e-2, strict_mode=False, max_iter=100):
    reward_matrix = np.expand_dims(reward_matrix, axis=1)
    for n in range(max_iter):
        print('iterating policy at ', n)
        old_best_a = np.argmax(Q, 1)
        v_pi = evaluate_policy(Q, transition_matrix, reward_matrix, gamma, theta)
        # assumes r = r(s) not r(s,a)
        # r(s,a)
        Q = reward_matrix + gamma * transition_matrix.dot(v_pi)
        best_a = np.argmax(Q, 1)
        if strict_mode:
            if np.all(best_a == old_best_a):
                print('policy stabilized at: ', n)
                break
        else:
           match_proportion = np.sum(best_a == old_best_a)/best_a.shape[0]
           if match_proportion > 0.99:
               break
            # the latter condition required to handle an edge case where there are ties
    return Q

def Q_value_iteration(transition_matrix, reward_matrix, theta=1e-3, gamma=0.99):
    '''
    a simplified version of Q value iteration
    reference: slide 9 of http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_6_value_functions.pdf
    '''
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]
    reward_matrix = np.expand_dims(reward_matrix, axis=1)
    v_old = np.zeros((num_states))
    for t in itertools.count():
        Q = reward_matrix + transition_matrix.dot(gamma * v_old)
        # if we want tie breaking
        #v = np.zeros(num_states)
        #for s in range(num_states):
        #    # evaluate this policy's action choices
        #    ties = np.flatnonzero(Q[s, :] == Q[s, :].max())
        #    a = np.random.choice(ties)
        #    v[s] = Q[s, a]
        v = np.max(Q, axis=1)
        #v[TERMINAL_STATE_ALIVE] = 0
        #v[TERMINAL_STATE_DEAD] = 0
        #Q[TERMINAL_STATE_ALIVE, :] = 0
        #Q[TERMINAL_STATE_DEAD, :] = 0
        max_delta= np.linalg.norm(v_old - v, np.inf)
        if max_delta < theta:
            #print('value converged after {} steps'.format(t))
            break
        v_old = v
    return Q


def solve_mdp(transition_matrix, reward_matrix, gamma=1.0):
    '''
    solve bellman equation by inverting matrix
    essentially finding a fixed point in one-step
    this approach does not work very well, though computationally fast
    hard to pin down why but it feels wrong right?
    '''
    #reward_matrix = [0 for i in range(81)]
    #reward_matrix[50] = 50
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]
    # to make transition_matrix compatible with reward function
    # we squash action dimension so T = s x s'
    transition_matrix_ss = np.sum(transition_matrix, axis=1)
    # solve bellman equation
    # A v = b
    A = np.identity(num_states) - gamma*transition_matrix_ss
    #A = (np.identity(transition_matrix_ss.shape[0]) - gamma*transition_matrix_ss)
    b = np.dot(transition_matrix_ss, reward_matrix)
    v_star = np.linalg.solve(A, b)
    # recover pi_star
    Q = compute_Q_from_v_star(v_star, transition_matrix, reward_matrix, gamma)
    pi = GreedyPolicy(num_states, num_actions, Q)
    #pi = EpsilonGreedyPolicy(num_states, num_actions, Q, epsilon=0.01)
    return pi
