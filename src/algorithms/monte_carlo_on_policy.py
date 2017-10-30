import numpy as np


def run_mc_on_policy(env, num_episodes, pi_behavior, gamma):
    # default pi_behavior -> epsilon greedy
    num_actions = env.num_actions
    num_states = env.num_states
    
    # keep track of these
    reward_sum = np.zeros((num_states, num_actions))
    visit_count = np.zeros((num_states, num_actions))
    D = []

    for episode_i in range(num_episodes):
        exps = run_mc_actor(env, pi_behavior)
        D.append(exps)
        run_mc_learner(exps, pi_behavior, reward_sum, visit_count, gamma)
    return D, pi_behavior


def run_mc_actor(env, pi_behavior, max_local_iter=200):
    exps = []

    env.reset()
    s = env.observe()
    probs = pi_behavior.query_Q_probs(s)
    a = np.random.choice(np.arange(len(probs)), p=probs)
    iter_i = 0
    while iter_i < max_local_iter:
        new_s, r = env.perform_action(a)
        #print('took {} at {} got {}'.format(a, s, r))
        probs = pi_behavior.query_Q_probs(new_s)
        new_a = np.random.choice(np.arange(len(probs)), p=probs)
        exps.append((s, a, r, new_s))
        if env.is_terminal(s):
            break
        else:
            s = new_s
            a = new_a
        iter_i += 1
        
        #if iter_i > MAX_ITER:
            # need termination if not promising
            # because mc has super high variance
            # break
    
    #print('Monte Carlo On Policy episode ended at ', iter_i)
    return exps


def run_mc_learner(exps, pi_behavior, reward_sum, visit_count, gamma):
    sa_pairs = set([(e[0], e[1]) for e in exps])
    for s, a in sa_pairs:
        first_occurrence_i = next(i for i, e in enumerate(exps) if e[0]==s and e[1]==a)
        rewards = [(gamma**i) * e[2] for i, e in enumerate(exps[first_occurrence_i:])]
        G = np.sum(rewards)
        reward_sum[s, a] += G
        visit_count[s, a] += 1.0
        avg_r = reward_sum[s, a] / visit_count[s, a]
        pi_behavior.update_Q(s, a, avg_r)
    
    return reward_sum, visit_count, pi_behavior

