import numpy as np
import numba as nb
import logging
import itertools

from policy.policy import EpsilonGreedyPolicy, GreedyPolicy
from learners.monte_carlo_on_policy import run_mc_actor

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
def iterate_policy(Q, transition_matrix, reward_matrix, gamma=0.99, theta=1e-2, strict_mode=False,
        max_iter=100):
    reward_matrix = np.expand_dims(reward_matrix, axis=1)
    for n in range(max_iter):
        print('iterating policy at ', n)
        old_best_a = np.argmax(Q, 1)
        v_pi = evaluate_policy(Q, transition_matrix, reward_matrix, gamma, theta)
        # assumes r = r(s) not r(s,a)
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
        Q = reward_matrix + gamma * transition_matrix.dot(v_old)
        v = np.max(Q, axis=1)
        max_delta= np.linalg.norm(v_old - v, np.inf)
        if max_delta < theta:
            #print('value converged after {} steps'.format(t))
            break
        v_old = v
    Q_star = reward_matrix + gamma * np.dot(transition_matrix, v)
    return Q_star


def solve_mdp(transition_matrix, reward_matrix, gamma=1.0):
    '''
    solve bellman equation by inverting matrix
    essentially finding a fixed point in one-step
    this approach does not work very well, though computationally fast
    hard to pin down why but it feels wrong right?
    '''
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]
    # to make transition_matrix compatible with reward function
    # we squash action dimension so T = s x s'
    transition_matrix_ss = np.sum(transition_matrix, axis=1)
    # solve bellman equation
    # A v = b
    A = np.identity(num_states) - gamma*transition_matrix_ss
    b = np.dot(transition_matrix_ss, reward_matrix)
    v_star = np.linalg.solve(A, b)
    # recover pi_star
    Q = compute_Q_from_v_star(v_star, transition_matrix, reward_matrix, gamma)
    pi = GreedyPolicy(num_states, num_actions, Q)
    #pi = EpsilonGreedyPolicy(num_states, num_actions, Q, epsilon=0.01)
    return pi
