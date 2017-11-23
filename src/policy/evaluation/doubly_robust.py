import numpy as np
from utils.utils import *
from mdp.dynamic_programming import evaluate_policy, evaluate_policy_Q


def eval_doubly_robust(env, D, pi_evaluation, gamma):
    DR = []
    num_actions = env.num_actions
    num_states = env.num_states
    esp = 1e-4
    RMIN = esp
    RMAX = 1.5
    build_approximate_model = make_approximate_model_builder(num_states, num_actions)
    
    for epi_i in range(len(D)):
        H_i, pi_behavior_i = D[epi_i]
        # using D until epi_i, build an approximate model
        T_hat, R_hat = build_approximate_model(H_i)
        # evalaute pi_e on our imperfect model
        # TODO: fix evaluate pi
        Q_hat_pi_e, V_hat_pi_e = evaluate_policy_Q(pi_evaluation, T_hat, R_hat, gamma)
        # compute importance weights
        X = 0.0
        Y = 0.0
        Z = 0.0
        DR_i = 0.0
        n = epi_i + 1
        rho = 1.0
        w_t = 1.0
        #import pdb;pdb.set_trace()
        for t, exp in enumerate(H_i):
            s, a, r, _ = exp
            # compute X: rewards scaled by importance weights
            p1 = pi_evaluation.query_Q_probs(s, a)
            p2 = pi_behavior_i[s, a]
            # compute importance ratio
            
            rho_t = (p1 + esp) / (p2 + esp)
            rho *= rho_t
            rho = np.clip(rho, RMIN, RMAX)
            w_t_min_1 = w_t
            w_t = rho / n

            X = (gamma ** t) * w_t * r
            #print('x', X)
            # compute Y: estimate of action value function
            Y = (gamma**t)* w_t * Q_hat_pi_e[s, a]
            #print('y', Y)
            # copmute Z: estimate of state value function
            Z = (gamma ** t) * w_t_min_1 * V_hat_pi_e[s]
            #print('z', Z)
            # using D until epi_i, we compute DR and log it here
            DR_i += X - Y + Z
        #print('DR of {} at {}th episode'.format(DR_i, epi_i))
        DR.append(DR_i) # notice DR is just V_hat_pi_e return DR 

    return DR

def make_approximate_model_builder(num_states, num_actions):
    transition_count_table = np.zeros((num_states, num_actions, num_states))
    reward_sum_table = np.zeros((num_states, num_actions))
 
    def build_approximate_model(episode):
        # replay experiences and build N_sas and R_sa
        for s, a, r, new_s in episode:
           transition_count_table[s, a, new_s] += 1
           reward_sum_table[s, a] += r

        # build T_hat and R_hat
        transition_matrix = np.zeros((num_states, num_actions, num_states))
        reward_table = np.zeros((num_states, num_actions))

        for s in range(num_states):
            for a in range(num_actions):
                N_sa = np.sum(transition_count_table[s, a, :])
                if N_sa == 0:
                    # if never visited, no reward, no transition
                    continue
                reward_table[s, a] = reward_sum_table[s, a] / N_sa
                for new_s in range(num_states):
                    N_sas = transition_count_table[s, a, new_s]
                    transition_matrix[s, a, new_s] = N_sas / N_sa
        
        return transition_matrix, reward_table
    
    return build_approximate_model

