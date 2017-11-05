import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import seaborn as sns
sns.set(style='dark')
from scipy.optimize.minpack import curve_fit

class QuadOpt():
    def __init__(self,epsilon = 0.01):
        self.SVMKernelType = "linear"
        self.mu_list = []
        self.target = None
        self.epsilon = epsilon
        self.SVMpenalty = 5500
        self.margins = []
        self.counter = 0

    def transform_data(self, target_mu, cur_mu ):
        self.mu_list.append(cur_mu)
        X = self.mu_list + [target_mu]
        X = np.array(X)
        y = np.zeros(len(X)).astype(np.int)
        y[-1] = 1
        y = np.array(y)
        return X, y
    
    def optimize(self, target_mu, cur_mu, norm_weights=True):
        clf = svm.SVC(kernel=self.SVMKernelType, C=self.SVMpenalty)
        X,y = self.transform_data(target_mu,cur_mu)
       
        clf.fit(X,y)
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        converged = (margin <= self.epsilon)
        weights = clf.coef_[0]
        bias = clf.intercept_

        self.margins.append(margin)
        self.counter += 1

        if self.counter % 10 == 0:
            fig, ax = plt.subplots()
            #fig, (ax1, ax2) = plt.subplots(1, 2)
            # plot mu vectors
            #x1 = np.linspace(X[:,0].min(), X[:,0].max(), 100)
            #x2 = -weights[0]/weights[1]*x1 - bias/weights[1]
            #c = range(len(y) - 1)
            #cm = plt.cm.get_cmap('Purples')
            #ax1.scatter(x=X[:-1, 0], y=X[:-1, 1], c=c, cmap=cm)
            #ax1.scatter(x=X[-1, 0], y=X[-1, 1], marker='*')
            #ax1.plot(x1, x2, label='hyperplane')
            # plot margins
            # curve fitting
            #exp_decay = lambda x, A, t, y0: A * np.exp(x * t) + y0
            #xx = range(self.counter)
            #params, cov = curve_fit(exp_decay, xx, self.margins, maxfev=10000)
            #yy = exp_decay(xx, *params)
            #ax2.plot(xx, yy, label='smooth')
            #ax2.plot(self.margins, label='margins')
            ax.plot(self.margins, label='margins')
            plt.legend()
            plt.savefig('margin{}'.format(self.counter), ppi=300, bbox_inches='tight')
        
        if norm_weights:
            weight_norm = np.linalg.norm(weights, 2)
            weights = weights/weight_norm
        else:
            weights = weights
        return weights, converged, margin
