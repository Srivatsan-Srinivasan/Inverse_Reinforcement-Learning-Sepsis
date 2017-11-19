import numpy as np
import numba as nb
import logging

from policy.policy import EpsilonGreedyPolicy, GreedyPolicy
# we need an efficient mdp solver

def iterate_value(Q_table, transition_matrix, reward_table, gamma=0.95, theta=0.1):
    num_states = Q_table.shape[0]
    num_actions = Q_table.shape[1]    
    # TODO: add utopia for value iteration?
    V = np.zeros(num_states)

    MAX_ITER = 50

    n = 0
    while n < MAX_ITER:
        n += 1
        delta = 0

        for s in range(num_states):
            old_v = V[s]
            Q_s = []
            # one-step look ahead            
            for a in range(num_actions):
                # update Q_table
                # if (s, a) unknown, this will be rmax
                Q_sa = 0
                for next_s in range(num_states):
                    p = transition_matrix[s, a, next_s]
                    if len(reward_table.shape) == 2:
                        # determinstic envirnment
                        r = reward_table[s, a]
                    else:
                        # stochastic environment
                        r = reward_table[s, a, next_s]
                    Q_sa += p * (r + gamma * V[next_s]) 
                Q_table[s, a] = Q_sa

            V[s] = np.max(Q_table[s, :])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < theta:
            break
    # print(n, 'terminated at value iteration steps')
    # print(np.sum(reward_table[reward_table < 0]), 'value iteration steps')

    return Q_table

@nb.jit
def evaluate_policy(Q, transition_matrix, reward_table, gamma=0.95, theta=1e-3, max_iter=100):
    num_states = Q.shape[0]
    v = np.zeros(num_states)
    for n in range(max_iter):
        v_temp = v
        for s in range(num_states):
            # evaluate this policy's action choices
            ties = np.flatnonzero(Q[s, :] == Q[s, :].max())
            a = np.random.choice(ties)
            r = reward_table[s]
            v[s] = transition_matrix[s, a, :].dot(r + gamma * Q[:, a])
        if np.abs(v_temp - v).max() < theta:
            print('policy evaluated after {} steps'.format(n))
            break
    return v


@nb.jit
def iterate_policy(Q, transition_matrix, reward_table, gamma=0.99, theta=1e-1,
        max_iter=100):
    for n in range(max_iter):
        print('iterating policy at ', n)
        # s x 1
        old_best_a = np.argmax(Q, 1)
        # s x 1
        v_pi = evaluate_policy(Q, transition_matrix, reward_table, gamma, theta)
        # s x a x s dot s x 1
        Q = transition_matrix.dot(reward_table + gamma * v_pi)
        # do policy improvement: update Q_table
        best_a = np.argmax(Q, 1)
        max_delta = np.abs(old_best_a - best_a).max()
        if np.all(best_a == old_best_a) or max_delta > theta:
            # the latter condition required to handle an edge case where there are ties
            print('policy stabilized at: ', n)
            break
        print('policy iteration broke after max_iter at: ', n)
    return Q


def compute_Q_from_v_star(v_star, transition_matrix, reward_matrix, gamma):
    # s x a = s x 1 + s x a x 1
    Q = np.expand_dims(reward_matrix, axis=1) + gamma * np.dot(transition_matrix, v_star)
    return Q

def solve_mdp(transition_matrix, reward_matrix, gamma=1.0):
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
    #pi = GreedyPolicy(num_states, num_actions, Q)
    pi = EpsilonGreedyPolicy(num_states, num_actions, Q, epsilon=0.01)
    return pi

