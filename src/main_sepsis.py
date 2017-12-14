from experiments.experiment import ExperimentManager, Experiment
from constants import *

import numpy as np
import argparse
import time

def get_arg_parser():
    parser = argparse.ArgumentParser(description='process configuration vars')
    # dev level
    parser.add_argument('-gnd', '--generate_new_data', action='store_true', dest='generate_new_data')
    parser.set_defaults(generate_new_data=False)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.set_defaults(verbose=False)
    parser.add_argument('-up', '--use_pca', action='store_true', dest='use_pca')
    parser.set_defaults(use_pca=False)
    parser.add_argument('-p', '--parallelized', action='store_true', dest='parallelized')
    parser.set_defaults(parallelized=False)
    parser.add_argument('-nt', '--num_trials', default=2, type=int, dest='num_trials')
    parser.add_argument('-ni', '--num_iterations', default=10, type=int, dest='num_iterations')

    parser.add_argument('-nb', '--num_bins', type=int, choices=[2, 4], dest='num_bins')
    # optimizer stuff
    parser.add_argument('-sp', '--svm_penalty', default=300.0, type=float, dest='svm_penalty')
    parser.add_argument('-se', '--svm_epsilon', default=0.0001, type=float, dest='svm_epsilon')
    parser.add_argument('-en', '--experiment_name', default='', type=str, help="name to be displayed in tensorboard", dest="experiment_name")
    parser.add_argument('-hm', '--hyperplane_margin', action='store_true', dest='hyperplane_margin')
    parser.set_defaults(hyperplane_margin=False)
    parser.add_argument('-net', '--num_expert_trajectories', default=1500, dest='num_exp_trajectories')
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    exps = []
    em = ExperimentManager(args)

    cur_t = time.strftime('%y%m%d_%H%M%S', time.gmtime())
#    exp1 = Experiment(
#        experiment_id =  cur_t + '_' + 'irl_greedy_physician_greedy',
#        policy_expert = em.pi_expert_phy_g,
#        save_file_name = cur_t + '_' + IRL_GREEDY_PHYSICIAN_Q_GREEDY ,
#        irl_use_stochastic_policy=False
#    )
#    em.set_experiment(exp1)
#
#    exp2 = Experiment(
#        experiment_id = cur_t + '_' + 'irl_greedy_physician_stochastic',
#        policy_expert = em.pi_expert_phy_s,
#        save_file_name = cur_t + '_' +  IRL_GREEDY_PHYSICIAN_Q_STOCHASTIC,
#        irl_use_stochastic_policy=False
#    )
#    em.set_experiment(exp2)
#
#    exp3 = Experiment(
#        experiment_id = cur_t + '_' + 'irl_greedy_mdp_greedy',
#        policy_expert = em.pi_expert_mdp_g,
#        save_file_name = cur_t + '_' + IRL_GREEDY_MDP_Q_GREEDY,
#        irl_use_stochastic_policy=False
#    )
#    em.set_experiment(exp3)
#
    #exp4 = Experiment(
    #    experiment_id = cur_t + '_' + 'irl_greedy_mdp_stochatic',
    #    policy_expert = em.pi_expert_mdp_s,
    #    save_file_name = cur_t + '_' + IRL_GREEDY_MDP_Q_STOCHASTIC,
    #    irl_use_stochastic_policy=False
    #)
    #em.set_experiment(exp4)

    exp5 = Experiment(
        experiment_id = cur_t + '_' + 'irl_stochastic_physician_greedy',
        policy_expert = em.pi_expert_phy_g,
        save_file_name = cur_t + '_' + IRL_STOCHASTIC_PHYSICIAN_Q_GREEDY,
        irl_use_stochastic_policy=True
    )
    em.set_experiment(exp5)

    exp6 = Experiment(
        experiment_id = cur_t + '_' + 'irl_stochatic_physician_stochastic',
        policy_expert = em.pi_expert_phy_s,
        save_file_name = cur_t + '_' + IRL_STOCHASTIC_PHYSICIAN_Q_STOCHASTIC,
        irl_use_stochastic_policy=True
    )
    em.set_experiment(exp6)

    #exp7 = Experiment(
    #    experiment_id = cur_t + '_' + 'irl_stochastic_mdp_greedy',
    #    policy_expert = em.pi_expert_mdp_g,
    #    save_file_name = cur_t + '_' + IRL_STOCHASTIC_MDP_Q_GREEDY,
    #    irl_use_stochastic_policy=True
    #)
    #em.set_experiment(exp7)

    exp8 = Experiment(
        experiment_id = 'irl_stochastic_mdp_stochatic',
        policy_expert = em.pi_expert_mdp_s,
        save_file_name = cur_t + '_' + IRL_STOCHASTIC_MDP_Q_STOCHASTIC,
        irl_use_stochastic_policy=True
    )
    em.set_experiment(exp8)

    em.run()


