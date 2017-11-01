import numpy as np
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

    
def evaluate_policy(pi, transition_matrix, reward_table, gamma=0.95, theta=1, max_iter=100):
    if isinstance(pi, np.ndarray):
        num_states, num_actions = pi.shape
    else:
        num_states, num_actions = pi.Q.shape
    V_pi = np.zeros(num_states)
    n = 0
    # if state space large, the state sweep gets really expensive
    while n < max_iter:
        n += 1
        delta = 0
        #print('iter setp', n)
        for s in range(num_states):
            # evaluate this policy's action choices                       
            temp = V_pi[s]
            V_pi[s] = 0
            if isinstance(pi, np.ndarray):
                ties = np.flatnonzero(pi[s, :] == pi[s, :].max())
                chosen_a = np.random.choice(ties)
            else:
                probs = pi.query_Q_probs(s)
                chosen_a = np.random.choice(np.arange(len(probs)), p=probs)

            for next_s in range(num_states):
                p = transition_matrix[s, chosen_a, next_s]
                if len(reward_table.shape) == 2:
                    # determinstic envirnment
                    r = reward_table[s, chosen_a]
                else:
                    # stochastic environment
                    r = reward_table[s, chosen_a, next_s]
                V_pi[s] += p * (r + gamma * V_pi[next_s])
            delta = max(delta, np.abs(temp - V_pi[s]))

        if delta < theta:
            break

    return V_pi

def evaluate_policy_Q(pi, transition_matrix, reward_table, gamma=0.95, theta=1, max_iter=100):
    #TODO: just a utility wrapper on evaluate_policy
    # this shall be merged into evlauate_policy
    if isinstance(pi, np.ndarray):
        num_states, num_actions = pi.shape
    else:
        num_states, num_actions = pi.Q.shape
    V_pi = np.zeros(num_states)
    Q_pi = np.zeros((num_states, num_actions))
    n = 0
    
    while n < max_iter:
        n += 1
        delta = 0

        for s in range(num_states):
            # evaluate this policy's action choices                       
            temp = V_pi[s]
            for a in range(num_actions):
                Q_pi[s, a] = 0
                for next_s in range(num_states):
                    p = transition_matrix[s, a, next_s] 
                    if len(reward_table.shape) == 2:
                        # determinstic envirnment
                        r = reward_table[s, a]
                    else:
                        # stochastic environment
                        r = reward_table[s, a,  next_s]
                    Q_pi[s, a] += p * (r + gamma * V_pi[next_s])
            V_pi[s] = np.max(Q_pi[s, :])
            delta = max(delta, np.abs(temp - V_pi[s]))
        print('policy evaluation', n)
        if delta < theta:
            break

    return Q_pi, V_pi


def iterate_policy(Q_table, transition_matrix, reward_table, gamma=0.95, theta=1,
        max_iter=100):
    # TODO: fix this if needed, and
    num_states = Q_table.shape[0]
    num_actions = Q_table.shape[1]
    
    n = 0 
    while n < max_iter:
        is_stable = True
        V_pi = evaluate_policy(Q_table, transition_matrix, reward_table, gamma, theta)
        n += 1
        print('poicy iteration n', n)
        for s in range(num_states):
            # for every state, compare cur_pi(s) and improved_pi(s)                  
            old_best_a = np.argmax(Q_table[s, :])
            Q_s = []
            
            # do policy improvement: update Q_table
            for a in range(num_actions):
                # print(transition_matrix[s, a, :])
                Q_sa = 0
                for next_s in range(num_states):
                    p = transition_matrix[s, a, next_s] 
                    if len(reward_table.shape) == 2:
                        # determinstic envirnment
                        r = reward_table[s, a]
                    else:
                        # stochastic environment
                        r = reward_table[s, a, next_s]
                    Q_sa += p * (r + gamma * V_pi[next_s]) 
                Q_table[s, a] = Q_sa

            best_a = np.argmax(Q_table[s, :])
            delta = np.abs(Q_table[s, best_a] - Q_table[s, old_best_a])
            if old_best_a != best_a and delta > theta:
                # the latter condition required to handle an edge case where there are ties
                is_stable = False
        
        if is_stable:
            break
    return Q_table


def solve_mdp(transition_matrix, reward_matrix):
    Q = np.zeros(reward_matrix.shape)
    Q_star = iterate_policy(Q, transition_matrix, reward_matrix)
    return Q_star


def solve_mdpr(transition_matrix, reward_matrix, gamma=0.99):
    num_states = transition_matrix.shape[0]
    # to make transition_matrix compatible with reward function
    # we squash action dimension so T = s x s'
    transition_matrix = np.sum(transition_matrix, axis=1)
    # solve bellman equation
    # A v = b
    A = (np.identity(transition_matrix.shape[0]) - gamma*transition_matrix)
    b = np.dot(transition_matrix, reward_matrix)

    v_star = np.linalg.solve(A, b)
    return v_star


def iterate_value_mdpr(Q_table, transition_matrix, compute_reward, gamma=0.99, theta=0.1):
    # too slow
    num_states = Q_table.shape[0]
    num_actions = Q_table.shape[1]    
    V = np.zeros(num_states)

    MAX_ITER = 50

    n = 0
    while n < MAX_ITER:
        n += 1
        delta = 0

        for s in range(num_states):
            old_v = V[s]
            Q_s = []
            print('at state', s)
            for a in range(num_actions):
                Q_sa = 0
                for next_s in range(num_states):
                    p = transition_matrix[s, a, next_s]
                    r = compute_reward(s)
                    Q_sa += p * (r + gamma * V[next_s]) 
                Q_table[s, a] = Q_sa

            V[s] = np.max(Q_table[s, :])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < theta:
            break
    # print(n, 'terminated at value iteration steps')
    # print(np.sum(reward_table[reward_table < 0]), 'value iteration steps')
    return Q_table
