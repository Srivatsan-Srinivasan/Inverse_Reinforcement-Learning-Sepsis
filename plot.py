import numpy as np
import scipy as sp

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()
# sns.set_palette("husl")

def calculate_interval(a, n):
    mu = np.mean(a)
    sem = sp.stats.sem(a)
    if sem == 0.0:
        # to prevent division by zero
        sem = np.finfo(float).eps
    return sp.stats.t.interval(0.80, n - 1, loc=mu, scale=sem)


def plot_arrow(location, direction, plot):
    arrow = plt.arrow(location[0], location[1], dx, dy, fc="k", ec="k", head_width=0.05, head_length=0.1)
    plot.add_patch(arrow) 


def _plot_iter_reward(reward_per_step, algo_name, line_color, line_style):
    mu = np.mean(reward_per_step, axis=0)
    x = np.arange(mu.shape[0])
    n = reward_per_step.shape[0]

    intervals = np.apply_along_axis(calculate_interval, 0, reward_per_step, n=n)
    plt.plot(mu, lw=2, color=line_color, linestyle=line_style, label=algo_name)
    plt.fill_between(x, intervals[0,:], intervals[1,:], alpha=.3, color=line_color)
    plt.title('') 
    plt.xlabel('iteration')
    plt.ylabel('cumulative reward throughout trial')
    plt.legend()


def _plot_episode_reward(reward_per_episode, algo_name, line_color, line_style):
    mu = np.mean(reward_per_episode, axis=0)
    x = np.arange(mu.shape[0])
    n = reward_per_episode.shape[0]
    intervals = np.apply_along_axis(calculate_interval, 0, reward_per_episode, n=n)
    plt.plot(mu, lw=2, color=line_color, linestyle=line_style, label=algo_name)
    plt.fill_between(x, intervals[0,:], intervals[1,:], alpha=.3, color=line_color)

    plt.title('') 
    plt.xlabel('episode')
    plt.ylabel('total reward per episode')
    plt.legend()




def _plot_Q_and_pi(maze, Q_table, algo_name):
    row_count = len(maze)
    col_count = len(maze[0]) 

    value_function = np.reshape(np.max(Q_table, 1), (row_count, col_count))
    policy_function = np.reshape(np.argmax(Q_table, 1), (row_count, col_count))
    wall_info = .5 + np.zeros((row_count, col_count))

    wall_mask = np.zeros((row_count , col_count) )
    for row in range(row_count):
        for col in range(col_count):
            if maze[row][col] == '#':
                wall_mask[row,col] = 1 
    wall_info = np.ma.masked_where( wall_mask==0 , wall_info )

    fig3 = plt.figure(figsize=(10,5))
    # value function plot 
    plt.imshow(value_function, interpolation='none', cmap=mpl.cm.jet)
    plt.colorbar()
    # plt.title('Policy for {}'.format(algo_name))        
    plt.imshow(wall_info, interpolation='none', cmap=mpl.cm.gray)
    # plt.imshow(1 - wall_mask, interpolation='none', cmap=mpl.cm.jet)
    
    for row in range(row_count):
        for col in range(col_count):
            if wall_mask[row][col] == 1:
                continue 
            if policy_function[row,col] == 0:
                dx = 0; dy = -.5
            if policy_function[row,col] == 1:
                dx = 0; dy = .5
            if policy_function[row,col] == 2:
                dx = .5; dy = 0
            if policy_function[row,col] == 3:
                dx = -.5; dy = 0
            plt.arrow(col, row, dx, dy, shape='full', fc='w', ec='w', length_includes_head=True, head_width=.2)
    plt.title('Policy for {}'.format(algo_name))        
    plt.show(fig3)


def plot_reward_per_iter(maze, maze_name, algorithms):
    # Useful stats for the plot
    fig1 = plt.figure(figsize=(10,5))
    plt.title(maze_name)
    for Q_table, reward_per_step, reward_per_episode, algo_name, ls, lc in algorithms:
        _plot_iter_reward(reward_per_step, algo_name, lc, ls)
    plt.show(fig1)


def plot_reward_per_episode(maze, maze_name, algorithms):
    # Useful stats for the plot
    fig2 = plt.figure(figsize=(10,5))
    plt.title(maze_name)
    for Q_table, reward_per_step, reward_per_episode, algo_name, ls, lc in algorithms:
        _plot_episode_reward(reward_per_episode, algo_name, lc, ls)
    plt.show(fig2)


def plot_policy_value_fn(maze, maze_name, algorithms):
    # Useful stats for the plot
    for Q_table, reward_per_step, reward_per_episode, algo_name, ls, lc in algorithms:
        _plot_Q_and_pi(maze, Q_table, algo_name)


def plot_cum_reward_action_error(data):
    summary = []
    action_error_probs = []
    for reward_per_step, reward_per_episode, algo_name, action_err_p in data:        
        reward_per_trial = np.sum(reward_per_episode, axis=1)
        summary.append(reward_per_trial)
        action_error_probs.append(action_err_p)

    fig = plt.figure(figsize=(10,5))
    plt.boxplot(summary, labels=action_error_probs)
    plt.title('Algorithm: {}'.format(algo_name))
    plt.xlabel('action error probability')
    plt.ylabel('total reward per trial')        
    plt.show(fig)


    
def plot_end_reward_action_error(data):
    summary = []
    action_error_probs = []
    for reward_per_step, _, algo_name, action_err_p in data:        
        end_rewards = reward_per_step[:, -100:]
        reward_per_trial = np.sum(end_rewards, axis=1)
        summary.append(reward_per_trial)
        action_error_probs.append(action_err_p)
    
    fig = plt.figure(figsize=(10,5))
    plt.boxplot(summary, labels=action_error_probs)
    plt.title('Algorithm: {}'.format(algo_name))
    plt.xlabel('action error probability')
    plt.ylabel('the last 100-step reward')        
    plt.show(fig)

