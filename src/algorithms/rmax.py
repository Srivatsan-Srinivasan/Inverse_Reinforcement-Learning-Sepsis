from value_iteration import VI, PI

import numpy as np

rmax_hyperparams_1 = {
    "algo_name": 'rmax_1',
    "gamma": .95,
    "visit_count": 5
}


rmax_hyperparams_2 = {
    "algo_name": 'rmax_2',
    "gamma": .95,
    "visit_count": 10
}

rmax_hyperparams_3 = {
    "algo_name": 'rmax_3', 
    "gamma": .95,
    "visit_count": 50
}

def greedy_policy(state, Q_table):
    # if there are ties
    ties = np.flatnonzero(Q_table[state, :] == Q_table[state, :].max())
    action = np.random.choice(ties)
    return action


def rmax_train(env, hyperparams, trial_count=1, episode_min_count=100, global_min_iter_count=5000, local_max_iter_count=500):        

    if 'visit_count' in hyperparams:
        MIN_VISIT_COUNT = hyperparams['visit_count']
    else:
        MIN_VISIT_COUNT = 10

    if 'gamma' in hyperparams:
        gamma = hyperparams['gamma']

    # add utopia state
    num_states = env.num_states + 1
    num_actions = env.num_actions
    rmax = env.get_max_reward()

    reward_per_episode = np.zeros((trial_count, episode_min_count))
    trial_lengths = []
    reward_per_step = np.zeros((trial_count, global_min_iter_count))
    
    for trial_idx in range(trial_count):

        # Initialize the Q table 
        Q_table = np.zeros((num_states, num_actions))
        mean_Q_table = np.zeros((num_states, num_actions))

        transition_count_table = np.zeros((num_states, num_actions, num_states))
        transition_matrix = np.zeros((num_states, num_actions, num_states))
        reward_table = np.zeros((num_states, num_actions, num_states))
        reward_sum_table = np.zeros((num_states, num_actions, num_states))
        
        # initialize utopia that does not exist but that's super awesome
        utopia_state = num_states - 1
        reward_table[utopia_state,:, utopia_state] = rmax
        # make utopia an absorbing state
        transition_matrix[:, :, utopia_state] = 1
        
        global_iter_idx = 0

        # Loop until the episode is done 
        for episode_idx in range(episode_min_count):
            # print('episode count {}'. format(episode_idx))
            # print('global iter count {}'. format(global_iter_idx))
            
            local_iter_idx = 0
            episode_reward_list = []

            # start the game
            env.reset()
            state = env.observe()
            action = greedy_policy(state, Q_table) 

            # Loop until done
            while True:
                # print('local iter count {}'. format(local_iter_idx))
                                             
                new_state, reward = env.perform_action(action)
                new_action = greedy_policy(new_state, Q_table)

                # increment for experiencing (s, a)
                transition_count_table[state, action, new_state] += 1
                reward_sum_table[state, action, new_state] += reward
                N_sa = np.sum(transition_count_table[state, action, :])
                


                if N_sa >= MIN_VISIT_COUNT:
                    # print("state {} action {} are known".format(state, action))
                    # update transition and reward with knowledge acquired so far
                    for next_s in range(num_states):
                        N_sas = transition_count_table[state, action, next_s]
                        if N_sas == 0:
                            transition_matrix[state, action, next_s] = 0
                            reward_table[state, action, next_s] = 0            
                        else:
                            transition_matrix[state, action, next_s] = N_sas / N_sa    
                            r_sum = reward_sum_table[state, action, next_s]
                            reward_table[state, action, next_s] = r_sum / N_sas
                    
                else:
                    # print("unknown state {} action {}".format(state, action))
                    reward_table[state, action, utopia_state] = rmax
                    # transition_matrix[state, action, utopia_state] = 1
                
                # stop if at goal/else update for the next iteration 
                if env.is_terminal(state):
                    break
                else:
                    state = new_state
                    action = new_action

                # update greedy_policy with new info
                if global_iter_idx % 50 == 0:
                    Q_table = PI(Q_table, transition_matrix, reward_table, gamma, theta=1)
                
                # store the data
                episode_reward_list.append(reward)
                if global_iter_idx < global_min_iter_count:
                    reward_per_step[trial_idx, global_iter_idx] = reward 
                
                local_iter_idx += 1
                global_iter_idx += 1
                
            # Store the rewards
            reward_per_episode[trial_idx, episode_idx] = np.sum(episode_reward_list)
        
        # trial ends
        mean_Q_table += Q_table
        trial_lengths.append(global_iter_idx)
        reward_per_step[trial_idx, :] = np.cumsum(reward_per_step[trial_idx, :])

    # slice off to the shortest trial for consistent visualization
    # reward_per_step = reward_per_step[:,:np.min(trial_lengths)]
    # shave off the paradise
    mean_Q_table = mean_Q_table[:utopia_state, :]
    mean_Q_table = mean_Q_table / trial_count
    
    return mean_Q_table, reward_per_step, reward_per_episode
