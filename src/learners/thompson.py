import numpy as np

from value_iteration import *

thompson_hyperparams_1 = {
    "algo_name": 'thompson_1', 
    "dirichlet_alpha": 1,
    "initial_reward": 50,
    "gamma": .95,
}

thompson_hyperparams_2 = {
    "algo_name": 'thompson_2', 
    "dirichlet_alpha": 1,
    "initial_reward": 10,
    "gamma": .95,
}

thompson_hyperparams_3 = {
    "algo_name": 'thompson_3',
    "dirichlet_alpha": 10,
    "initial_reward": 50,
    "gamma": .95,
}



def policy(state, Q_table, action_count, epsilon, iter_idx):
    epsilon = np.cbrt(epsilon / (iter_idx + 1))
    probs = np.ones(action_count, dtype=float) * epsilon / action_count
    best_action = np.argmax(Q_table[state, : ])
    probs[best_action] += 1. - epsilon
    action = np.random.choice(np.arange(action_count), p=probs)
    return action


def greedy_policy(state, Q_table):
    # if there are ties
    ties = np.flatnonzero(Q_table[state, :] == Q_table[state, :].max())
    action = np.random.choice(ties)
    return action

def sample_MDP(transition_count_table, reward_table, Dirichlet_alpha):
    # FILL THIS IN 
    # R has 3 categories (goal+moved, hit_wall, pitfall+moved)
    # we assume P(r|H) ~ Dir(goal+moved, hit_wall, pitfall+moved)
    # s has num_state categories
    # we assume P(s|H) ~ Dir((s1,a1)(s1,a2) ... (sm,an))
    num_states = transition_count_table.shape[0]
    num_actions = transition_count_table.shape[1]
    transition_matrix = np.zeros((num_states, num_actions, num_states))
    
    # priors = np.zeros(num_states)
    # for s in range(num_states):
    #     priors = Dirichlet_alpha + transition_count_table[s,:]
    #     transition_matrix[s, :] = np.random.dirichlet(priors)
    for s in range(num_states):
        for a in range(num_actions):
            priors =  Dirichlet_alpha + transition_count_table[s, a, :]
            # size = 1 is too high variance
            posterior = np.mean(np.random.dirichlet(priors, size=10), axis=0)
            # most_frequent = np.argmax(posterior)
            # if np.sum(transition_count_table[s, a, :]) > 5:
            #     posterior = np.eye(num_states)[most_frequent]
            transition_matrix[s, a, :] = posterior
    return transition_matrix, reward_table

def thompson_train(env, hyperparams, trial_count=1, episode_min_count=100, global_min_iter_count=5000, local_max_iter_count=100):
    
    num_states = env.num_states
    num_actions = env.num_actions

    Dirichlet_alpha = hyperparams['dirichlet_alpha']
    initial_reward = hyperparams['initial_reward']
    gamma = hyperparams['gamma']

    reward_per_episode = np.zeros((trial_count, episode_min_count))
    reward_per_step = np.zeros((trial_count, global_min_iter_count)) 
    trial_lengths = []

    for trial_idx in range(trial_count):

        # Initialize the Q table 
        Q_table = np.zeros((num_states, num_actions))
        mean_Q_table = np.zeros((num_states, num_actions))

        global_iter_idx = 0
      
        transition_count_table = np.zeros((num_states, num_actions, num_states))
        reward_table = np.zeros((num_states, num_actions, num_states))
        reward_table.fill(initial_reward)


        # Loop until the episode is done 
        for episode_idx in range(episode_min_count):
            # print('episode count {}'. format(episode_idx))
            # print('global iter count {}'. format(global_iter_idx))
            
            local_iter_idx = 0
            episode_reward_list = []

            # start the game
            env.reset()
            state = env.observe()
            # action = greedy_policy(state, Q_table)

            action = policy(state, Q_table, num_actions, 1, local_iter_idx)
            transition_matrix, reward_table = sample_MDP(transition_count_table, reward_table, Dirichlet_alpha)


            # Loop until done
            while True:
                # print('local iter count {}'. format(local_iter_idx))
                                             
                new_state, reward = env.perform_action(action)
                # print(state, action, reward, new_state)
                # new_action = greedy_policy(new_state, Q_table)
                new_action = policy(state, Q_table, num_actions, 0.5, local_iter_idx)
                # increment for experiencing (s, a, s')
                # increment unit of 100 to give more mass
                transition_count_table[state, action, new_state] += 1
                # assumg reward is deterministic

                reward_table[state, action, new_state] = reward
                # stop if at goal/else update for the next iteration 
                if env.is_terminal(state):
                    break
                else:
                    state = new_state
                    action = new_action

                if global_iter_idx % 50 == 0:
                    # update greedy_policy with new info
                    transition_matrix, reward_table = sample_MDP(transition_count_table, reward_table, Dirichlet_alpha)
                    Q_table = VI(Q_table, transition_matrix, reward_table, gamma, theta=initial_reward/5)

                episode_reward_list.append(reward)
                if global_iter_idx < global_min_iter_count:
                    reward_per_step[trial_idx, global_iter_idx] = reward

                local_iter_idx += 1
                global_iter_idx += 1
            
            # episode ends
            reward_per_episode[trial_idx, episode_idx] = np.sum(episode_reward_list)

        # trial ends
        mean_Q_table += Q_table
        trial_lengths.append(global_iter_idx)        
        reward_per_step[trial_idx, :] = np.cumsum(reward_per_step[trial_idx, :])

    # slice off to the shortest trial for consistent visualization
    reward_per_step = reward_per_step[:,:np.min(trial_lengths)]
    mean_Q_table = mean_Q_table / trial_count

    return mean_Q_table, reward_per_step, reward_per_episode
