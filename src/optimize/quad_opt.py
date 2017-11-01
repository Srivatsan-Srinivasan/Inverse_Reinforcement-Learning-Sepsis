import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

class QuadOpt():
    def __init__(self,epsilon = 0.01):
        self.SVMKernelType = "linear"
        self.mu_list = []
        self.epsilon = epsilon
        self.SVMpenalty = 1000
    
    def transform_data(self, target_mu, cur_mu ):
        self.mu_list.append(cur_mu) 
        X_candidate = self.mu_list
        X_target = target_mu        
        y = [0 for i in range(len(X_candidate))]
        X_candidate.append(X_target)
        self.mu_list.append(X_target)
        y.append(1)
        return X_candidate,y
    
    def optimize(self, target_mu, cur_mu, norm_weights = True):
        clf = svm.SVC(kernel= self.SVMKernelType, C= self.SVMpenalty)
        
        X,y = self.transform_data(target_mu,cur_mu)
       
        clf.fit(X,y)
        margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
        converged = (margin <= self.epsilon)
        weights = clf.coef_
        pdb.set_trace()
        if norm_weights:
            weight_norm = sum([abs(w) for w in weights.flatten()])
            weights = weights.flatten()/weight_norm
        return weights, converged
      
        
