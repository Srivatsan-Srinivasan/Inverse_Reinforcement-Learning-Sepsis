import numpy as np
from utils.utils import *

def eval_weighted_importance_sampling(env, D, pi_evaluation, gamma):
    WIS = []
    # implement incremental weighted importance sampling
    # to lower the variance but not unbiased
    # but this has asymtotically unbiased (==consistent)
    esp = 1e-4
    RMIN = esp
    RMAX = 1.5
    num_actions = env.num_actions
    num_states = env.num_states
    Q_hat = np.zeros((num_states, num_actions))
    # for importance sampling
    C = np.zeros((num_states, num_actions))
    env.reset()
    s_origin = env.observe()
    for epi_i in range(len(D)):
        H_i, pi_behavior_i = D[epi_i]
        W_t = 1.0
        G = 0.0
        for s, a, r, new_s in reversed(H_i):
            G = gamma * G + r
            C[s, a] += W_t
            # improving Q_hat here
            Q_hat[s, a] += (W_t / C[s,a]) * (G - Q_hat[s,a])
            # TODO: add tie breaking
            p1 = pi_evaluation.query_Q_probs(s, a)
            # TODO: hate the notational incosistency
            p2 = pi_behavior_i[s, a]
            if p1 == 0.0 or p2 == 0.0:
                break
            #rho = (p1+1e-4) / (p2+1e-4)
            rho = p1 / p2
            rho = np.clip(rho, RMIN, RMAX)
            W_t *= rho
            W_t = np.clip(W_t, RMIN, RMAX)
        WIS.append(G)
    return WIS


