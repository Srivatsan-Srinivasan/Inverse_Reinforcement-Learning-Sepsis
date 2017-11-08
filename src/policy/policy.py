import numpy as np
import numba as nb

class EpsilonGreedyPolicy:
    def __init__(self, num_states, num_actions, Q=None, epsilon=0.01, theta=1e-2, decay_rate=1e-4):
        if Q is None:
            self._Q = np.zeros((num_states, num_actions))
        else:
            self._Q = Q
        self._eps = epsilon
        self._theta = theta
        self._decay_rate = decay_rate
        self._min_eps = 1e-2

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)
    
    def query_Q_probs(self, s=None, a=None):
        if s is None and a is None:
            num_states, num_actions = self._Q.shape
            Q_probs = np.zeros((num_states, num_actions))
            for s in range(num_states):
                Q_probs[s, :] = self._query_Q_probs(s)
            return Q_probs

            return np.copy(self._Q_probs)
        else:
            return self._query_Q_probs(s, a)

    def update_Q_probs(self, s, a, val):
        self._Q[s,a] = val
    
    def _query_Q_probs(self, s, a=None):
        # TODO: fix the terminology... prob vs. non-prob
        num_actions = self._Q.shape[1]
        probs = np.ones(num_actions, dtype=float) * self._eps / num_actions
        # there's a subtle bug that occurs without this...
        #ties = np.isclose(self._Q[s, :], self._Q[s, :].max(), self._theta)
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        #ties = np.flatnonzero(ties)
        #print(self._Q[s,:])
        if a is None:
            # we want the probs of pi(s, .)
            best_a = np.random.choice(ties)
            probs[best_a] += 1. - self._eps
            return probs
        else:
            # we want the prob of pi(s,a)
            if a in ties:
                probs[a] += 1. - self._eps
            return probs[a]

    def query_Q_val(self, s, a=None):
        # TODO: some bound checks
        if a is None:
            return self._Q[s, :]
        else:
            return self._Q[s, a]
    
    def update_epsilon(self):
        self._eps -= self._decay_rate
        self._eps = np.max([self._eps, self._min_eps])

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
        if s is None and a is None:
            num_states, num_actions = self._Q.shape
            Q_probs = np.zeros((num_states, num_actions))
            for s in range(num_states):
                Q_probs[s, :] = self._query_Q_probs(s)
            return Q_probs

            return np.copy(self._Q_probs)
        else:
            return self._query_Q_probs(s, a)
    
    def update_Q_probs(self, s, a, val):
        self._Q[s,a] = val
    
    def _query_Q_probs(self, s, a=None):
        num_actions = self._Q.shape[1]
        ties = np.flatnonzero(self._Q[s, :] == self._Q[s, :].max())
        best_a = np.random.choice(ties)
        probs = np.eye(num_actions, dtype=float)[best_a] 
        if a is None:
            return probs
        else:
            return probs[a]

    def query_Q_val(self, s, a=None):
        # TODO: some bound checks
        if a is None:
            return self._Q[s, :]
        else:
            return self._Q[s, a]


class RandomPolicy:
    def __init__(self, num_states, num_actions):
        self._Q = np.zeros((num_states, num_actions))
        self._Q_probs = np.ones((num_states, num_actions), dtype=float) / num_actions

    @property
    def Q(self):
        # support read-only
        return np.copy(self._Q)


    def query_Q_probs(self, s=None, a=None):
        if s is None and a is None:
            # support read-only
            return np.copy(self._Q_probs)
        elif a is None:
            return self._Q_probs[0, :]
        else:
            return self._Q_probs[0, a]

    def update_Q_probs(self, s, a, val):
        # do nothing...
        pass

def make_fast_random_policy(num_states, num_actions):
    Q = np.ones((num_states, num_actions), dtype=np.float32) / num_actions
    @nb.jit
    def f(s):
        return Q[s, :]
    return f

def make_fast_greedy_policy(num_states, num_actions, Q=None):
    if Q is None:
        Q = np.zeros((num_states, num_actions))
    @nb.jit
    def f(s):
        ties = np.flatnonzero(Q[s, :] == Q[s, :].max())
        best_a = np.random.choice(ties)
        probs = np.eye(num_actions, dtype=np.float32)[best_a]
        return probs
    return f

