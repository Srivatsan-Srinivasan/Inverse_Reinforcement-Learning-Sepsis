from mdp.builder import make_mdp
from mdp.solver import solve_mdpr
from policy.policy import GreedyPolicy
from policy.custom_policy import get_physician_policy
from utils.utils import load_data, extract_trajectories
from irl.irl import *
from optimize.quad_opt import QuadOpt
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
sns.set(style='dark', palette='husl')

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
    trajectories = extract_trajectories(df_cleansed, NUM_PURE_STATES)
    transition_matrix, reward_matrix = make_mdp(trajectories, NUM_STATES, NUM_ACTIONS)
    
    df_centroids = df_centroids[['SOFA', 'age']]
    # arbitrary feature columns to use
    # they become binary arbitrarily
    # to check how, see phi() definition
    feature_columns = df_centroids.columns
    
    # initialize s_0 sampler
    sample_initial_state = make_initial_state_sampler(df_cleansed)
    get_state = make_state_centroid_finder(df_centroids, feature_columns)
    phi = make_phi(df_centroids)
    
    # initialize w
    np.random.seed(1)
    alphas = np.ones(len(feature_columns))
    W = np.random.dirichlet(alphas, size=1)[0]
    
    # get pi_expert
    pi_expert = get_physician_policy(trajectories)
    mu_pi_expert = estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, phi, pi_expert)
    
    v_pi_expert = estimate_v_pi(W, mu_pi_expert)
    
    # initialize opt
    opt = QuadOpt(0.01)
    # we don't need this part for now, but let's keep it just in case
    # initialize with a Greedy Policy
    # we can swap for other types of pis later
    # we may have to index s.t. pi_tilda_i
    #pi_tilda = GreedyPolicy(NUM_STATES, NUM_ACTIONS)
    #mu_pi_tilda = estimate_feature_expectation(transition_matrix, sample_initial_state, get_state, pi_tilda)
    #v_pi_tilda = estimate_v_pi(W, mu_pi_tilda)
    margins = []
    num_episodes = 100
    l2_norms = np.zeros(num_episodes)

    for epi_i in range(num_episodes):
        # solve mdp with r=w phi to get pi_tilda
        compute_reward = make_reward_computer(W, get_state, phi)
        reward_matrix = np.asarray([compute_reward(s) for s in range(NUM_STATES)])
        pi_tilda = solve_mdpr(transition_matrix, reward_matrix)
        mu_pi_tilda = estimate_feature_expectation(transition_matrix,
                                                   sample_initial_state,
                                                   get_state, phi, pi_tilda)

        v_pi_tilda = estimate_v_pi(W, mu_pi_tilda)
        # diff
        l2_norm = np.linalg.norm(mu_pi_tilda - mu_pi_expert, 2)
        l2_norms[epi_i] = l2_norm
        W, converged, margin = opt.optimize(mu_pi_expert, mu_pi_tilda)
        margins.append(margin)
        #print('rolling avg. margin', avg_margin[-1])
        
        #print('vpi_tild - vpi_expert', diff)
        if converged:
            print('converged!!!')
            break
    plt.figure(figsize=(10, 10))
    plt.plot(l2_norms)
    plt.show()
    import pdb;pdb.set_trace()
