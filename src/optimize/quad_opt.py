import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class QuadOpt():
    def __init__(self,epsilon = 0.01):
        self.SVMKernelType = "linear"
        self.mu_list = []
        self.target = None
        self.epsilon = epsilon
        self.SVMpenalty = 0.5
        plt.ion()
        self.is_tmu = True
        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax
        self.plot = ax.scatter([], [])
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)

    def transform_data(self, target_mu, cur_mu ):
        self.mu_list.append(cur_mu)
        X = self.mu_list + [target_mu]
        y = np.zeros(len(X)).astype(np.int)
        y[-1] = 1

        arr = self.plot.get_offsets()
        if self.is_tmu:
            array = np.append(arr, cur_mu)
            self.ax.scatter(target_mu[0], target_mu[1], marker='*', color='y')
            self.is_tmu = False
        else:
            array = np.append(arr, cur_mu)
        self.plot.set_offsets(array)
        # update x and ylim to show all points:
        self.ax.set_xlim(array.min() - 10, array.max() + 10)
        self.ax.set_ylim(array.min() - 10, array.max() + 10)

        # update the figure
        self.fig.canvas.draw()
        
        return X, y
    
    def optimize(self, target_mu, cur_mu, norm_weights=True):
        #import pdb;pdb.set_trace()
        clf = svm.SVC(kernel=self.SVMKernelType, C=self.SVMpenalty)
        X,y = self.transform_data(target_mu,cur_mu)
       
        clf.fit(X,y)
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        converged = (margin <= self.epsilon)
        weights = clf.coef_
        import pdb;pdb.set_trace()
        if norm_weights:
            weight_norm = np.linalg.norm(weights, 2)
            weights = weights.flatten()/weight_norm
        else:
            weights = weights.flatten()
        return weights, converged, margin
