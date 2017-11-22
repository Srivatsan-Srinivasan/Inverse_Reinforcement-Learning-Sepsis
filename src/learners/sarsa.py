import numpy as np

def get_alpha(params, iter_idx):
    if params['algo_name'] in ['sarsa_1', 'q_1']:
        return 1 / iter_idx
    elif params['algo_name'] in ['sarsa_2', 'q_2']:
        return np.min([.5, 10 / iter_idx])
    elif params['algo_name'] in ['sarsa_3', 'q_3']:
        return .1
    else:
        print("oops, something's wrong with alpha..")



sarsa_hyperparams_1 = {
    "algo_name": 'sarsa_1',
    "alpha": get_alpha,
    "epsilon": .1,
    "gamma": .95,
}

sarsa_hyperparams_2 = {
    "algo_name": 'sarsa_2', 
    "alpha": get_alpha, # special handling
    "epsilon": .1,
    "gamma": .95,
}


sarsa_hyperparams_3 = {
    "algo_name": 'sarsa_3', 
    "alpha": get_alpha,
    "epsilon": .1,
    "gamma": .95,
}

def policy(state, Q_table, action_count, epsilon):
    probs = np.ones(action_count, dtype=float) * epsilon / action_count
    best_action = np.argmax(Q_table[state, : ])
    probs[best_action] += 1. - epsilon
    action = np.random.choice(np.arange(action_count), p=probs)
    return action


# Update the Q table 
def update_Q_SARSA(Q_table, alpha, gamma, state, action, reward, new_state, new_action):
    # new_action ~ pi(new_state)    
    td_target = reward + gamma * Q_table[new_state, new_action]
    td_delta = td_target - Q_table[state, action]
    Q_table[state, action] += alpha * td_delta
    return Q_table 



def sarsa_train(env, hyperparams, trial_count=1, episode_min_count=100, global_min_iter_count=5000, local_max_iter_count=100):        
    # Loop over some number of episodes
    state_count = env.num_states
    action_count = env.num_actions
    reward_per_episode = np.zeros((trial_count, episode_min_count))
    reward_per_step = np.zeros((trial_count, global_min_iter_count)) 
    # episode gets terminated when past local_max

    if 'epsilon' in hyperparams:
        epsilon = hyperparams['epsilon']

    if 'gamma' in hyperparams:
        gamma = hyperparams['gamma']

    trial_lengths = []
    for trial_idx in range(trial_count):

        # Initialize the Q table 
        Q_table = np.zeros((state_count, action_count))
        transition_count_table = np.zeros((state_count, state_count))
        reward_value_table = np.zeros((state_count)) 
        
        global_iter_idx = 0

        # Loop until the episode is done 
        for episode_idx in range(episode_min_count):
            # print('episode count {}'. format(episode_idx))
            # print('global iter count {}'. format(global_iter_idx))
            
            local_iter_idx = 0
            # Start the env 
            env.reset()
            state = env.observe()
            action = policy(state, Q_table, action_count, epsilon) 
            episode_reward_list = []

            # Loop until done
            while local_iter_idx < local_max_iter_count:
                # print('local iter count {}'. format(local_iter_idx))
                                
                new_state, reward = env.perform_action(action)
                new_action = policy(new_state, Q_table, action_count, epsilon) 

                # FILL IN HERE: YOU WILL NEED CASES FOR THE DIFFERENT ALGORITHMS
                
                if 'alpha' in hyperparams:
                    alpha = hyperparams['alpha'](hyperparams, np.cbrt(global_iter_idx+1))

                Q_table = update_Q_SARSA(Q_table, alpha, gamma, state, action, reward, new_state, new_action)
                
                # store the data
                episode_reward_list.append(reward)
                if global_iter_idx < global_min_iter_count:
                    reward_per_step[trial_idx, global_iter_idx] = reward

                # stop if at goal/else update for the next iteration 
                if env.is_terminal(state):
                    break
                else:
                    state = new_state
                    action = new_action
                
                local_iter_idx += 1
                global_iter_idx += 1

            # Store the rewards
            reward_per_episode[trial_idx, episode_idx] = np.sum(episode_reward_list)

        trial_lengths.append(global_iter_idx)
        reward_per_step[trial_idx, :] = np.cumsum(reward_per_step[trial_idx, :])

    # slice off to the shortest trial for consistent visualization
    reward_per_step = reward_per_step[:,:np.min(trial_lengths)]
    return Q_table, reward_per_step, reward_per_episode

