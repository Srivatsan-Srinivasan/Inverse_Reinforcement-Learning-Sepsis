from experiments.experiment import ExperimentManager, Experiment
from constants import *

import numpy as np
import argparse

def get_arg_parser():
    parser = argparse.ArgumentParser(description='process configuration vars')
    # dev level
    parser.add_argument('-gnd', '--generate_new_data', action='store_true', dest='generate_new_data')
    parser.set_defaults(generate_new_data=False)
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose')
    parser.set_defaults(verbose=False)
    parser.add_argument('-up', '--use_pca', action='store_true', dest='use_pca')
    parser.set_defaults(use_pca=False)
    parser.add_argument('-nt', '--num_trials', default=2, type=int, dest='num_trials')
    parser.add_argument('-ni', '--num_iterations', default=10, type=int, dest='num_iterations')

    parser.add_argument('-nb', '--num_bins', type=int, choices=[2, 4], dest='num_bins')
    # optimizer stuff
    parser.add_argument('-sp', '--svm_penalty', default=300.0, type=float, dest='svm_penalty')
    parser.add_argument('-se', '--svm_epsilon', default=0.0001, type=float, dest='svm_epsilon')
    parser.add_argument('-en', '--experiment_name', default='', type=str, help="name to be displayed in tensorboard", dest="experiment_name")
    return parser

if __name__ == '__main__':
    args = get_arg_parser().parse_args()
    exps = []
    em = ExperimentManager(args)

    exp1 = Experiment(
        experiment_id ='greedy_physician',
        policy_expert =em.pi_expert_phy_g,
        save_file_name = IRL_PHYSICIAN_Q_GREEDY,
        irl_use_stochastic_policy=False
    )
    em.set_experiment(exp1)

    exp2 = Experiment(
        experiment_id = 'stochastic_physician',
        policy_expert = em.pi_expert_phy_s,
        save_file_name = IRL_PHYSICIAN_Q_STOCHASTIC,
        irl_use_stochastic_policy=False
    )
    em.set_experiment(exp2)

    exp3 = Experiment(
        experiment_id = 'greedy_mdp',
        policy_expert = em.pi_expert_mdp_g,
        save_file_name = IRL_MDP_Q_GREEDY,
        irl_use_stochastic_policy=False
    )
    em.set_experiment(exp3)

    exp4 = Experiment(
        experiment_id = 'stochastic_mdp',
        policy_expert = em.pi_expert_mdp_s,
        save_file_name = IRL_MDP_Q_STOCHASTIC,
        irl_use_stochastic_policy=False
    )
    em.set_experiment(exp4)

    em.run()


