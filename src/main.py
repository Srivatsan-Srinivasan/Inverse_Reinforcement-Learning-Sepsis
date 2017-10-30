from mdp.builder import make_mdp
from mdp.solver import solve_mdp
from policy.policy import GreedyPolicy
from utils.utils import load_data, get_physician_policy
from irl.irl import *

# let us think about what we need purely
'''
- MDP: states, transitions, reward(just in case)
- phi, weights
- policy
- mdp solver: given T and R(phi*W), find the policy that minimizes the expected difference

todo
- make mdp builder work
    - fix transition matrix not summing to one
    - add load and save state centroids
    - find binary cols
- get the mvp irl workflow done
    - implement estimate feature expectation
    - implement naive phi
    - implement reward function
    - implement state value estimation function
- get expert pi_e
- test if the mdp solver work
- make mdp more efficienct (using outside code)
'''

if __name__ == '__main__':
    # loading the whole data
    # TODO: load only train data
    df, df_cleansed, df_centroids = load_data()
    transition_matrix, reward_matrix = make_mdp(df_cleansed, NUM_STATES, NUM_ACTIONS)
    
    # arbitrary feature columns to use
    # they become binary arbitrarily
    # to check how, see phi() definition
    feature_columns = ['1','2', '3']
    
    # initialize s_0 sampler
    sample_initial_state = make_initial_state_sampler(df_cleansed)
    get_state = make_state_centroid_finder(df_centroids, feature_columns)
    
    # initialize w
    np.random.seed(1)
    alphas = np.ones(len(feature_columns))
    W = np.random.dirichlet(alphas, size=1)[0]
    
    # initialize with a Greedy Policy
    # we can swap for other types of pis later
    # we may have to index s.t. pi_tilda_i
    pi_tilda = GreedyPolicy(NUM_STATES + 2, NUM_ACTIONS)
    mu_pi_tilda = estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, pi_tilda)
    v_pi_tilda = estimate_v_pi(W, mu_pi_tilda)


