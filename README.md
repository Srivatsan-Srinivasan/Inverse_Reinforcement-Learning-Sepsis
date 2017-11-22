## TODOs
- testing on gridworld. reproduce similar results as Abbeel.
- implement policy extraction from a set of policies found in IRL (at the end of Section 3 in Abbeel (2004)
- hyperparameter tuning (especially with C in SVM)
- testing

## Potential TODOs
- max margin planner (Ratliff, 2004): worth noting because we can specify some custom loss function using empirically safe actions over some prespecified states like high variance states (from sepsis.csv) Since we already have max margin learner, implementing this should not take too much time.
- max entropy learner (Ramachandran, 2007): worth noting because this can encode uncertainty in the derived reward function (the ill-posed nature of the IRL). can also account for expert suboptimality.

## Implemented
- [ ] plots from testing on gridworld (margin/dist_mu vs. number of iterations)
- [ ] plots from testing on gridworld (v_pi vs. number of trajectories)
- [ ] plots from testing on sepsis (margin/dist_mu vs. number of iterations)
- [ ] plots from testing on sepsis (v_pi vs. number of trajectories)
- [ ] plots from varying expert policy (deterministic/stochastic) on gridworld/sepsis 
- [x] max margin learner (Abbeel, 2004)
- [ ] max margin planner (Ratliff, 2004) 
- [ ] max entropy learner (Ramachandran, 2007)

## examples
### against empirical expert policy (clincian)
 <img alt='reward margin' src="/src/img/exp_margin_i15.png" width="500" height="500">
 <img alt='diff in mu' src="/src/img/exp_dist_mu_i15.png" width="500" height="500">
 <img alt='diff in v_pi' src="/src/img/exp_v_pi_i15.png" width="500" height="500">
### against artificial expert policy (optimal mdp solution)
 <img alt='reward margin' src="/src/img/exp2_margin_i15.png" width="500" height="500">
 <img alt='diff in mu' src="/src/img/exp2_dist_mu_i15.png" width="500" height="500">
 <img alt='diff in v_pi' src="/src/img/exp2_v_pi_i15.png" width="500" height="500">

## gettings started
'''
cd src/
python main.py
'''
