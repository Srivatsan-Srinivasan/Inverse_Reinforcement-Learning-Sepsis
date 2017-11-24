import numpy as np
import numba as nb

class EpsilonGreedyPolicy:
    '''
    TODO: refactor this 
    '''
    def __init__(self, num_states, num_actions, Q=None, epsilon=0.01):
        if Q is None:
            self._Q = np.zeros((num_states, num_actions))
        else:
            self._Q = Q
        self._eps = epsilon

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)
    
    def query_Q_probs(self, s=None, a=None):
        Q_probs = np.zeros(self._Q.shape)
        for s in range(self._Q.shape[0]):
            Q_probs[s, :] = self.query_Q_probs(s)
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]

    
    def _query_Q_probs(self, s, a=None):
        num_actions = self._Q.shape[1]
        probs = np.ones(num_actions, dtype=float) * self._eps / num_actions
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        if a is None:
            best_a = np.random.choice(ties)
            probs[best_a] += 1. - self._eps
            return probs
        else:
            if a in ties:
                probs[a] += 1. - self._eps
            return probs[a]

    def choose_action(self, s):
        probs = self._query_Q_probs(s)
        return np.random.choice(len(probs), p=probs)

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val

class GreedyPolicy:
    def __init__(self, num_states, num_actions, Q=None):
        if Q is None:
            # start with random policy
            self._Q = np.zeros((num_states, num_actions))
        else:
            # in case we want to import e-greedy
            self._Q = Q

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)
    
    def query_Q_probs(self, s=None, a=None):
        Q = self._Q - np.min(self._Q, axis=1)
        Q_probs = Q / np.sum(Q, axis=1)
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]

    def choose_action(self, s):
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        return np.random.choice(ties)
    
    def get_opt_actions(self):
        opt_actions = np.zeros(len(self._Q))
        for i in range(len(opt_actions)):
            opt_actions[i] = self.choose_action(i)
        return opt_actions
    
    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class StochasticPolicy:
    def __init__(self, num_states, num_actions, Q=None):
        if Q is None:
            # start with random policy
            self._Q = np.zeros((num_states, num_actions))
        else:
            # in case we want to import e-greedy
            # make Q non negative to be useful as probs
            self._Q = Q
        #self.opt_actions = np.ones((num_states, num_actions)) * -1

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def query_Q_probs(self, s=None, a=None, laplacian_smoothing=True):
        '''
        returns:
            probability distribution of actions over all states
        '''
        if laplacian_smoothing:
            LAPLACIAN_PROB = 0.01
            L = np.max(self._Q, axis=1) * LAPLACIAN_PROB
            Q = self._Q - np.min(self._Q, axis=1) + L
            #probs = np.random.dirichlet(alphas, size=1)[0]
            Q_probs = Q / np.sum(Q, axis=1)
        else:
            Q_probs = self._Q - np.min(self._Q, axis=1)
        if s is None and a is None:
            return Q_probs
        elif a is None:
            return Q_probs[s, :]
        else:
            return Q_probs[s, a]
    
    # L - Laplace smoother
    #def get_stochastic_actions(self, laplacian_smoothing=False):
    #        alphas = self._Q[s,:] - np.min(self._Q[s, :]) + L
    #        probs = np.random.dirichlet(alphas, size=1)[0]
    #        self.opt_actions[s] = probs
    #    return self.opt_actions

    #L - Laplace Smoother to have non-zero probability.
    def choose_action(self, s, laplacian_smoothing=False):
        probs = self.query_Q_probs(s, laplacian_smoothing=laplacian_smoothing)
        return np.random.choice(len(probs), p=probs)

    def update_Q_val(self, s, a, val):
        self._Q[s,a] = val


class RandomPolicy:
    def __init__(self, num_states, num_actions):
        self._Q_probs = np.ones((num_states, num_actions), dtype=float) / num_actions

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)

    def choose_action(self, s):
        probs = self._Q_probs[s, :]
        return np.random.choice(len(probs), p=probs)
