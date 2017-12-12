import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import sem
sns.set(style='dark', palette='husl')
from gridworld.constants_gridworld import IMG_PATH


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


def plot_margin_expected_value(margins, num_trials, num_iterations, plot_prefix='new'):
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
                 yerr=margin_se, fmt='-o')
    plt.xticks(np.arange(0, num_iterations+1, 5))
    plt.xlabel('Number of iterations', fontsize = 18)
    plt.ylabel('Margin in Expected Value', fontsize = 18)
    plt.legend(fontsize = 12)
    plt.savefig('{}{}_margin_t{}_i{}'.format(IMG_PATH, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()



def plot_diff_feature_expectation(dist_mus, num_trials, num_iterations, plot_prefix='new'):
    '''
    plot l2 distance between mu_expert and mu_pi_tilda
    '''
    dist_se = sem(dist_mus, axis=0)
    avg_dist_mus = np.mean(dist_mus, axis=0)
    fig = plt.figure(figsize=(10, 10))
    plt.ylim((0, np.max(dist_mus) * 1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_dist_mus,
                 label=r'$||\mu_E-\mu_{\tilde{\pi}}||$',
                 yerr=dist_se, fmt='-o')
    plt.xticks(np.arange(1, num_iterations+1, 5))
    plt.xlabel('Number of iterations', fontsize = 18)
    plt.ylabel('L2 Distance in Feature Expectation', fontsize = 18)
    plt.legend(fontsize = 12)
    plt.savefig('{}{}_dist_mu_t{}_i{}'.format(IMG_PATH, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()



def plot_value_function(v_pis, v_pi_expert, num_trials, num_iterations, plot_prefix='new'):
    avg_v_pis = np.mean(v_pis, axis=0)
    v_pi_se = sem(v_pis, axis=0)
    fig = plt.figure(figsize=(10, 10))
    plt.ylim((np.min(avg_v_pis)*.8, np.max(avg_v_pis)*1.2))
    plt.errorbar(np.arange(1, num_iterations+1), avg_v_pis, yerr=v_pi_se,
                 fmt='-o', label=r'$E_{s_0 \sim D(s)}[V^{\tilde \pi}(s_0)]$')
    plt.axhline(v_pi_expert, label=r'$E_{s_0 \sim D(s)}[V^{\pi_E}(s_0)]$', c='c')
    plt.xticks(np.arange(1, num_iterations+1, 5))
    plt.xlabel('Number of iterations')

    plt.ylabel('Performance')
    plt.legend()
    plt.savefig('{}{}_v_pi_t{}_i{}'.format(IMG_PATH, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()

def plot_intermediate_rewards():
    '''
    TODO: need to be implemented
    '''
    pass

def plot_ir_and_policy(task_name, Q_table, reward_matrix, num_trials, num_iterations, plot_prefix='new'):
    # reward_matrix[-1] = 0

    # Useful stats for the plot
    row_count = len( task_name )
    col_count = len( task_name[0] ) 
    value_function = np.reshape( np.max( Q_table , 1 ) , ( row_count , col_count ) )
    policy_function = np.reshape( np.argmax( Q_table , 1 ) , ( row_count , col_count ) )
    reward_matrix = np.reshape( reward_matrix , ( row_count , col_count ) )

    # wall_info = .5 + np.zeros( ( row_count , col_count ) )
    # wall_mask = np.zeros( ( row_count , col_count ) )
    # for row in range( row_count ):
    #     for col in range( col_count ):
    #         if task_name[row][col] == '#':
    #             wall_mask[row,col] = 1     
    # wall_info = np.ma.masked_where( wall_mask==0 , wall_info )

    # # Plot the rewards
    fig = plt.figure(figsize=(10, 8))
    plt.subplot( 1 , 2 , 1 ) 
    plt.imshow( reward_matrix , interpolation='none' , cmap=plt.cm.jet )
    plt.title( 'Intermediate reward', fontsize = 20) 
    plt.colorbar()

    # value function plot 
    plt.subplot( 1 , 2 , 2 ) 
    plt.imshow( value_function , interpolation='none' , cmap=plt.cm.jet )
    plt.colorbar()
    # plt.imshow( wall_info , interpolation='none' , cmap=matplotlib.cm.gray )
    # plt.title( 'Value Function' )

    # policy plot 
    # plt.imshow( 1 - wall_mask , interpolation='none' , cmap=matplotlib.cm.gray )    
    for row in range( row_count ):
        for col in range( col_count ):
            # if wall_mask[row][col] == 1:
            #     continue 
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow( col , row , dx , dy , shape='full', fc='w' , ec='w' , lw=3, length_includes_head=True, head_width=.2 )
    plt.title( 'IRL policy', fontsize = 20) 
    plt.savefig('{}{}_policy_t{}_i{}'.format(IMG_PATH, plot_prefix, num_trials, num_iterations), ppi=300, bbox_inches='tight')
    plt.close()       
    # plt.show( block=False ) 
