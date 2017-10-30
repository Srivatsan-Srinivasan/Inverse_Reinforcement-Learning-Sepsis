from mdp.builder import make_mdp
from mdp.solver import solve_mdp
from policy.policy imoprt get_physician_policy
from irl.irl import *

# let us think about what we need purely
'''
- MDP: states, transitions, reward(just in case)
- phi, weights
- policy
- mdp solver: given T and R(phi*W), find the policy that minimizes the expected difference

todo
- get the mvp irl workflow done
- test if the mdp solver work
- make mdp more efficienct (using outside code)
'''



if __name__ == '__main__':
    import pdb;pdb.set_trace()
    df, state_centroids, transition_matrix, reward_matrix = make_mdp(num_states=750, num_actions=25)
    
    print('let us rock and roll')
    # find binary cols
	variables_to_use = find_binary_columns(df)
	# display binary columns
	df.iloc[:5,variables_to_use]
	gamma = 0.95
	action_count = 25
	state_count = 750

	variables_to_use = find_binary_columns(df)
	variable_count = len(variables_to_use)

	# initialize w
	w = np.zeros((variable_count))
	# initialize policy pi
	pi = np.zeros((state_count))
	# make some random centroid matrix
	centroid = np.ones((state_count, df.shape[1]))

	R = reward(w, centroid, variables_to_use)

	# sample trajectories
	m=100
	sample_trajectories = sampling_trajectories(transition_matrix, pi, m, state_count)
	mu = feature_expectation(sample_trajectories, gamma)

