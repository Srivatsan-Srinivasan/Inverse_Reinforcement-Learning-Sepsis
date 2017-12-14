## Max Margin IRL in Sepsis

## Getting Started

```python
# define your experiment in main_sepsis.py
# e.g.

exp1 = Experiment(
    experiment_id =  cur_t + '_' + 'irl_greedy_physician_greedy',
    policy_expert = em.pi_expert_phy_g,
    save_file_name = cur_t + '_' + IRL_GREEDY_PHYSICIAN_Q_GREEDY ,
    irl_use_stochastic_policy=False
)
em.set_experiment(exp1)

# run the following script in src/ directory
# e.g.
python main_sepsis.py -p -nt 5 -ni 15 -nb 2 -cm 'kp' -ns 100 -gnd
```
### Available arguments to the module
```
usage: main_sepsis.py [-h] [-gnd] [-v] [-up] [-cm {km,kp}] [-ns NUM_STATES]
                      [-p] [-nt NUM_TRIALS] [-ni NUM_ITERATIONS] [-nb {2,4}]
                      [-sp SVM_PENALTY] [-se SVM_EPSILON]
                      [-en EXPERIMENT_NAME] [-hm] [-net NUM_EXP_TRAJECTORIES]

process configuration vars

optional arguments:
  -h, --help            show this help message and exit
  -gnd, --generate_new_data
  -v, --verbose
  -up, --use_pca
  -cm {km,kp}, --clustering_method {km,kp}
                        kmeans or kprototype (cao, huang)
  -ns NUM_STATES, --num_states NUM_STATES
  -p, --parallelized
  -nt NUM_TRIALS, --num_trials NUM_TRIALS
  -ni NUM_ITERATIONS, --num_iterations NUM_ITERATIONS
  -nb {2,4}, --num_bins {2,4}
  -sp SVM_PENALTY, --svm_penalty SVM_PENALTY
  -se SVM_EPSILON, --svm_epsilon SVM_EPSILON
  -en EXPERIMENT_NAME, --experiment_name EXPERIMENT_NAME
                        name to be displayed in tensorboard
  -hm, --hyperplane_margin
  -net NUM_EXP_TRAJECTORIES, --num_expert_trajectories NUM_EXP_TRAJECTORIES
```

### Note
The module requires you have necessary data (Sepsis.csv) available in data/ directory.

### Experimental features
- K prototype clustering
