import numpy as np
from sklearn.svm import LinearSVC


class QuadOpt():
    def __init__(self, epsilon=0.01, penalty=1.0):
        self.mus = []
        self.epsilon = epsilon
        self.penalty = penalty

    def transform_data(self, target_mu, cur_mu):
        X = np.array(self.mus + [target_mu])
        y = np.empty(len(X))
        y.fill(-1)
        # target mu is labeled +1
        y[-1] = 1
        return X, y
    
    def optimize(self, target_mu, cur_mu, normalize=True):
        self.mus.append(cur_mu)
        X,y = self.transform_data(target_mu, cur_mu)
        clf = LinearSVC(C=self.penalty)
        clf.fit(X,y)
        # since decision hyperplane is W^T mu = 0
        # coefficients is a normal vector to hyperplane
        W = clf.coef_[0]
        norm = np.linalg.norm(W, 2)
        # TODO: I think this is wrong
        # margin = 1 / weight_norm
        if normalize:
            W = W / norm
        # taken from Abbeel (2004)
        # dist from a support vector to mu_expert
        diffs = target_mu - np.array(self.mus)
        # TODO: check if abs can be applied
        # otherwise margin can be negative
        margin = np.abs(W.dot(diffs.T)).min()
        converged = (margin < self.epsilon)
        return W, converged, margin

