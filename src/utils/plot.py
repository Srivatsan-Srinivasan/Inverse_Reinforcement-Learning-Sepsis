import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='dark', palette='Set1')
from scipy.stats import sem
from constants import IMG_PATH, NUM_PURE_STATES, NUM_ACTIONS
FONT_SIZE = 20
font = {'weight' : 'bold',
        'size'   : FONT_SIZE}
matplotlib.rc('font', **font)
np.set_printoptions(precision=6)

def plot_hyperplane(X, xx, yy):
    '''
    TODO: fix this
    plot hyperplane and vectors (mus)
    '''
    pass

def plot_margin_expected_value(margins, num_trials, num_iterations, save_path, plot_prefix='new'):
    '''
    plot margin in expected value of expert (best) and second-best policy
    params:
        margins: {num_trials x num_iterations} array containing margins
    '''
    avg_margins = np.mean(margins, axis=0)
    margin_se = sem(margins, axis=0)
    fig = plt.figure(figsize=(10, 10))
    plt.ylim((0, np.max(margins) * 1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_margins,
                 label=r'$w^T\mu_E-w^T\mu_{\tilde{\pi}}$',
                 yerr=margin_se, fmt='-o', lw=3)
    plt.xticks(np.arange(0, num_iterations+1, 5))
    plt.tick_params(labelsize=FONT_SIZE)
    plt.xlabel('Number of iterations', fontsize=FONT_SIZE)
    plt.ylabel('Margin in Expected Value', fontsize=FONT_SIZE)
    plt.legend()
    plt.savefig('{}{}_margin_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

def plot_diff_feature_expectation(dist_mus, num_trials, num_iterations, save_path, plot_prefix='new'):
    '''
    plot l2 distance between mu_expert and mu_pi_tilda
    '''
    dist_se = sem(dist_mus, axis=0)
    avg_dist_mus = np.mean(dist_mus, axis=0)
    fig = plt.figure(figsize=(10, 10))
    plt.ylim((0, np.max(dist_mus) * 1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_dist_mus,
                 label=r'$||\mu_E-\mu_{\tilde{\pi}}||$',
                 yerr=dist_se, fmt='-o', lw=3)
    plt.tick_params(labelsize=FONT_SIZE)
    plt.xticks(np.arange(1, num_iterations+1, 5))
    plt.xlabel('Number of iterations', fontsize=FONT_SIZE)
    plt.ylabel('L2 Distance in Feature Expectation', fontsize=FONT_SIZE)
    plt.legend()
    plt.savefig('{}{}_dist_mu_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

def plot_value_function(v_pis, v_pi_expert, num_trials, num_iterations, save_path, plot_prefix='new'):
    # performance relative to expert
    v_pis /= v_pi_expert
    # becomes one
    v_pi_expert /= v_pi_expert
    avg_v_pis = np.mean(v_pis, axis=0)
    v_pi_se = sem(v_pis, axis=0)
    fig = plt.figure(figsize=(10, 10))
    min_ylim = np.min([np.min(avg_v_pis), v_pi_expert])
    max_ylim = np.max([np.max(avg_v_pis), v_pi_expert])
    plt.ylim((min_ylim*.8, max_ylim*1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_v_pis, yerr=v_pi_se,
                 fmt='-o', label=r'$E_{s_0 \sim D(s)}[V^{\tilde \pi}(s_0)]$', lw=3)
    plt.axhline(v_pi_expert, label=r'$E_{s_0 \sim D(s)}[V^{\pi_E}(s_0)]$', c='c', lw=3)
    plt.xticks(np.arange(1, num_iterations+1, 5))
    plt.tick_params(labelsize=FONT_SIZE)
    plt.xlabel('Number of iterations', fontsize=FONT_SIZE)

    plt.ylabel('Performance', fontsize=FONT_SIZE)
    plt.legend()
    plt.savefig('{}{}_v_pi_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

def plot_performance_vs_trajectories(save_path,
                                     plot_prefix,
                                     num_trials,
                                     num_iterations,
                                     v_pi_irl_gs,
                                     v_pi_irl_ss,
                                     v_pi_expert):
    '''
    TODO: need to be implemented
    '''
    import pdb;pdb.set_trace()
    v_pi_irl_gs /= v_pi_expert
    v_pi_irl_ss /= v_pi_expert
    fig, ax = plt.subplot(figsize=(10,10))
    ax.plot(v_pi_irl_g, label='greedy')
    ax.plot(v_pi_irl_s, label='stochastic')
    ax.axhline(v_pi_expert, label='expert', c='c', lw=3)
    ax.set_xlabel('Number of Expert Trajectories')
    ax.set_ylabel('Value of Policy')
    ax.legend(loc='best')
    fig.savefig('{}{}_performance_vs_traj_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

def plot_deviation_from_experts(sd,
                                pi_phy,
                                pi_mdp_probs,
                                pi_irl_probs,
                                save_path,
                                plot_prefix,
                                num_trials,
                                num_iterations):
    '''
    compare clinician, mdp
    compare clinician, irl
    '''
    # hyperparameters
    # some preprocessing, cap bin jump by one bin only
    # e.g. for iv, std is too high, it often jumps by two bins
    thresholds = np.array([0.20, 0.30, 0.50])
    pi_phy_probs = pi_phy.query_Q_probs()
    violations = np.zeros((2, len(thresholds), NUM_PURE_STATES))
    num_bins = 5
    bins = np.arange(num_bins)
    for i, pi in enumerate([pi_mdp_probs, pi_irl_probs]):
        for j, th in enumerate(thresholds):
            for s in range(NUM_PURE_STATES):
                mode_action = np.argmax(pi_phy_probs[s, :])
                # @hack @todo fix this better
                mask = np.ones(num_bins, dtype=bool)
                aa = sd[sd[:, 1] == mode_action]
                #for k, _ in enumerate(aa):
                #    mask[np.unique(aa[k])] = 0
                # @hack
                mask[np.unique(aa[0])] = 0
                no_sigma_action_idx = bins[mask]
                # not included in -sigma, +sigma of mode
                num_violations = np.sum(pi[s, no_sigma_action_idx] > th)
                violations[i, j, s] = 0 if len(no_sigma_action_idx) == 0 else num_violations / len(no_sigma_action_idx)
            violations[i, j] = 100 * np.array(sorted(violations[i, j], reverse=True))

    fig= plt.figure(figsize=(10,10))
    linestyles = ['-',':']
    desc = ['mdp','irl']
    # @refactor
    colors = [['firebrick', 'red', 'darksalmon'], ['blue', 'royalblue', 'navy']]
    for i, _ in enumerate([pi_mdp_probs, pi_irl_probs]):
        for j, th in enumerate(thresholds):
            plt.plot(violations[i, j, :], linestyle=linestyles[i], c=colors[i][j], label='{}, threshold={:.2f}%'.format(desc[i], 100.0 * th))
    quartiles = np.arange(5)/4.0
    plt.xticks(np.around(750 * quartiles), 100 * quartiles)
    plt.legend(loc='best')
    plt.xlabel('State Quantiles (%)', size=FONT_SIZE)
    plt.ylabel('Proportion of Violations (%)', size=FONT_SIZE)
    fig.savefig('{}{}_violations_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

#def plot_deviation_from_experts(pi_phy,
#                                pi_phy_sd_l,
#                                pi_phy_sd_r,
#                                pi_mdp_probs,
#                                pi_irl_probs,
#                                save_path,
#                                plot_prefix,
#                                num_trials,
#                                num_iterations):
#    '''
#    compare clinician, mdp
#    compare clinician, irl
#    '''
#    # hyperparameters
#    # some preprocessing, cap bin jump by one bin only
#    # e.g. for iv, std is too high, it often jumps by two bins
#
#    thresholds = np.arange(0.05, 0.16, 0.05)
#    pi_phy_probs = pi_phy.query_Q_probs()
#    pi_phy_sd_l_probs = pi_phy_sd_l.query_Q_probs()
#    pi_phy_sd_r_probs = pi_phy_sd_r.query_Q_probs()
#    concat = np.array((pi_phy_probs, pi_phy_sd_r_probs, pi_phy_sd_r_probs))
#    pi_phy_probs_avg = np.mean(concat, axis=0)
#
#    violations = np.zeros((2, len(thresholds), NUM_PURE_STATES))
#    import pdb;pdb.set_trace()
#    for i, pi in enumerate([pi_mdp_probs, pi_irl_probs]):
#        for j, th in enumerate(thresholds):
#            for s in range(NUM_PURE_STATES):
#                low_p_action_idx = np.flatnonzero(pi_phy_probs_avg[s,:] <= th)
#                num_violations = np.sum(pi_irl_probs[s, low_p_action_indices] > th)
#                violations[i, j, s] = 0 if len(low_p_action_indices) == 0 else num_violations / len(low_p_action_indices)
#                violations[i, j, s] = 0 if len(low_p_action_indices) == 0 else num_violations / len(low_p_action_indices)
#            violations[i, j] = 100 * np.array(sorted(violations[i, j], reverse=True))
#            violations[i, j] = 100 * np.array(sorted(violations[i, j], reverse=True))
#
#    fig= plt.figure(figsize=(10,10))
#    for i, th in enumerate(thresholds):
#        plt.plot(violations[i, :], label='threshold: {:.2f}%'.format(100.0 * th))
#    quartiles = np.arange(5)/4.0
#    plt.xticks(np.around(750 * quartiles), quartiles)
#    plt.legend(loc='best')
#    plt.xlabel('States', size=FONT_SIZE)
#    plt.ylabel('Proportion of Violations (%)', size=FONT_SIZE)
#    fig.savefig('{}{}_violations_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
#    plt.close()

def plot_intermediate_rewards_vs_mortality(intermediate_rewards,
                                           avg_mortality_per_state,
                                           save_path,
                                           plot_prefix,
                                           num_trials,
                                           num_iterations):
    fig = plt.figure(figsize=(10,10))
    # select only high mortality states
    avg_mortality_per_state = np.array(avg_mortality_per_state)
    intermediate_rewards = intermediate_rewards[np.flatnonzero(avg_mortality_per_state > 70)]
    high_mortality_states = avg_mortality_per_state[np.flatnonzero(avg_mortality_per_state > 70)]
    sc = plt.scatter(np.arange(len(intermediate_rewards)), sorted(intermediate_rewards, reverse=True), c=high_mortality_states, cmap='Reds', label='hello world')
    plt.axhline(1, c='b', lw=3, linestyle='--')
    plt.axhline(-1, c='b', lw=3, linestyle='--')
    plt.legend(loc='best')
    plt.xlabel('States')
    plt.ylabel('Intermediate Rewards')
    plt.colorbar(sc)
    fig.savefig('{}{}_intermediate_rewards_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()


