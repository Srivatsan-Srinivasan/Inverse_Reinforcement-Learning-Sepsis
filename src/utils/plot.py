import numpy as np
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem
sns.set(style='dark', palette='Set1')
from constants import IMG_PATH
FONT_SIZE = 20
font = {'weight' : 'bold',
        'size'   : FONT_SIZE}
matplotlib.rc('font', **font)

def plot_hyperplane(X, xx, yy):
    '''
    TODO: fix this
    plot hyperplane and vectors (mus)
    '''
    pass
    #fig, ax = plt.subplots()
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    #x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
    #x2 = -weights[0]/weights[1]*x1 - bias/weights[1]
    #c = range(len(y) - 1)
    #cm = plt.cm.get_cmap('Purples')
    #ax1.scatter(x=X[:-1, 0], y=X[:-1, 1], c=c, cmap=cm)
    #ax1.scatter(x=X[-1, 0], y=X[-1, 1], marker='*')
    #ax1.plot(x1, x2, label='hyperplane')
    #exp_decay = lambda x, A, t, y0: A * np.exp(x * t) + y0
    #xx = range(self.counter)
    #ax2.plot(xx, yy, label='smooth')


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
    plt.xlabel('Number of iterations',fontsize=FONT_SIZE)
    plt.ylabel('L2 Distance in Feature Expectation', fontsize=FONT_SIZE)
    plt.legend()
    plt.savefig('{}{}_dist_mu_t{}xi{}'.format(save_path, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()



def plot_value_function(v_pis, v_pi_expert, num_trials, num_iterations, save_path, plot_prefix='new'):
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

def plot_intermediate_rewards():
    '''
    TODO: need to be implemented
    '''
    pass
