import numpy as np
import numba as nb
from policy.policy import EpsilonGreedyPolicy, GreedyPolicy
# we need an efficient mdp solver
def policy( state , Q_table , action_count , epsilon ):
    # an epsilon-greedy policy
    if np.random.random() < epsilon:
        action = np.random.choice( action_count ) 
    else: 
        action = np.argmax( Q_table[ state , : ] ) 
    return action 

# Update the Q table 
def update_Q_Qlearning( Q_table, state , action , reward , new_state , new_action, alpha=0.5, gamma=0.95 ):
    new_action = np.argmax(Q_table[state, :])
    Q_table[state, action] = Q_table[state, action] + alpha*(reward + gamma*Q_table[new_state, new_action]- Q_table[state, action])
    # FILL THIS IN 
    return Q_table 

def Q_learning_solver_for_irl(task, transition_matrix, reward_matrix, NUM_STATES, NUM_ACTIONS, episode_count = 500, max_task_iter = np.inf, epsilon = 0.2):
    # Initialize the Q table 
    Q_table = np.zeros( ( NUM_STATES , NUM_ACTIONS ) )

    # Initialize transition count table
    # transition_count_table = np.zeros((state_count, action_count, state_count))
    iteration = 0
    # temp_policy = np.zeros((NUM_STATES, NUM_ACTIONS))
    # optimal_policy = np.ones(temp_policy.shape)

    # while np.sum(temp_policy-optimal_policy) != 0:
    #     optimal_policy = np.copy(temp_policy)

    # Loop until the episode is done 
    for episode_iter in range( episode_count ):
        if iteration >= 5000 and episode_iter >= 100:
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
                # t_probs = np.copy(transition_matrix[state, action, :])
                # new_state = np.random.choice(NUM_STATES, p=t_probs)
                # reward = reward_matrix[state]
                new_action = policy( new_state , Q_table , NUM_ACTIONS , epsilon ) 
                
                # update transition table
                # transition_count_table[state, action, new_state] +=1
                # store the data
                iteration += 1
                    

                Q_table = update_Q_Qlearning(Q_table , 
                                             state , action , reward , new_state , new_action)

                # for state in range(NUM_STATES):
                #     ind_max_a = np.argmax(Q_table[state, :])    
                #     temp_policy[state, ind_max_a] = 1    

                # stop if at goal/else update for the next iteration 
                if task.is_terminal( state ):
                    break
                else:
                    state = new_state
                    action = new_action 
                   
    # derive optimal policy
    optimal_policy = GreedyPolicy(NUM_STATES, NUM_ACTIONS, Q_table)
    return optimal_policy, Q_table

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


def solve_mdp_iterate(transition_matrix, reward_matrix):
    Q = np.zeros(reward_matrix.shape)
    Q_star = iterate_policy(Q, transition_matrix, reward_matrix)
    return Q_star

@nb.jit(nopython=True)
def compute_Q_from_v_star(v_star, transition_matrix, reward_matrix, gamma):
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]
    Q = np.zeros((num_states, num_actions))
    for s in range(num_states):
        for a in range(num_actions):
            Q[s, a] = reward_matrix[s] + gamma * np.dot(transition_matrix[s, a], v_star)
    return Q

def solve_mdp(transition_matrix, reward_matrix, gamma=1.0):
    reward_matrix = [0 for i in range(81)]
    reward_matrix[50] = 50
    num_states = transition_matrix.shape[0]
    num_actions = transition_matrix.shape[1]
    # to make transition_matrix compatible with reward function
    # we squash action dimension so T = s x s'
    transition_matrix_ss = np.sum(transition_matrix, axis=1)
    # solve bellman equation
    # A v = b

    A = (np.identity(transition_matrix_ss.shape[0]) - gamma*transition_matrix_ss)
    b = np.dot(transition_matrix_ss, reward_matrix)
    v_star = np.linalg.solve(A, b)
    # recover pi_star
    Q = compute_Q_from_v_star(v_star, transition_matrix, reward_matrix, gamma)
    pi = GreedyPolicy(num_states, num_actions, Q)
    # pi = EpsilonGreedyPolicy(num_states, num_actions, Q, epsilon=0.01)s
    return pi


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
