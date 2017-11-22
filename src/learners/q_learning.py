import numpy as np
from policy.policy import GreedyPolicy

def Q_train(env, num_episodes, alpha, gamma, pi_behavior):
    num_states = env.num_states
    num_actions = env.num_actions

    # D = (H_i, pi_b_i)
    D = []
    episode_lengths = []

    for epi_i in range(num_episodes):
        env.reset()
        H_i = []

        s = env.observe()
        probs = pi_behavior.query_Q_probs(s)
        a = np.random.choice(np.arange(len(probs)), p=probs)
        iter_i = 0 
        
        while True:
            # actor
            new_s, r = env.perform_action(a)
            # print('took {} at {} got {}'.format(a, s, r))
            # store exp to D
            exp = (s, a, r, new_s)
            H_i.append(exp)

            # choose next action
            probs = pi_behavior.query_Q_probs(s)
            new_a = np.random.choice(np.arange(len(probs)), p=probs)
            
            # update learner
            run_Q_learner(pi_behavior, alpha, gamma, s, a, r, new_s)
            
            if env.is_terminal(s):
                break
            else:
                s = new_s
                a = new_a
            iter_i += 1

        pi_b_i = pi_behavior.query_Q_probs()
        D_i = (H_i, pi_b_i)
        D.append(D_i)
        episode_lengths.append(iter_i)

    #print(np.mean(episode_lengths))
    pi_target = GreedyPolicy(num_states, num_actions, pi_behavior.Q)
    return D, pi_target


def run_Q_learner(pi_behavior, alpha, gamma, s, a, r, new_s):
    Q_vals = pi_behavior.query_Q_val(new_s)
    ties = np.flatnonzero(Q_vals == Q_vals.max())
    best_a = np.random.choice(ties)

    td_target = r + gamma * pi_behavior.query_Q_val(new_s, best_a)
    td_delta = td_target - pi_behavior.query_Q_val(s, a)
    new_Q = pi_behavior.query_Q_val(s, a) + alpha * td_delta
    pi_behavior.update_Q_probs(s, a, new_Q)

