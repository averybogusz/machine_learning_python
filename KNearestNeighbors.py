import numpy as np
import matplotlib.pyplot as plt

def euclidean_norm(x, y, *args):
    return np.sum(((x - y) ** args[0]), axis =1)

def uniform_weights(dists, *args): #
    K = len(dists)
    return np.array([1/K]*K)

def reciprocal_weights(dists, *args):
    return 1/(np.sqrt(dists) + args[0])

def gauss_weights(dists, *args):
        ed = np.exp(dists)
        return ed/np.sum(ed)

def accuracy(y, y_hat):
    return np.mean(y == y_hat)

def R2(y, y_hat):
    return 1 - np.sum(( y - y_hat)**2)/(np.sum( (y-y.mean() )**2))
class KNN():
    def __init__(self, K, norm_func = euclidean_norm, weight_function = reciprocal_weights, weight_pars = 1e-2, norm_pars = 2, mode = 0): # this tells you when you define class it needs initial attributes
        self.mode = mode
        self.weight_pars = weight_pars
        self.norm_pars = norm_pars
        self.K = K
        self.weight_function = weight_function
        self.norm_func = norm_func


    def fit(self, X, y):
        self.X = X
        self.Y = y

    def  predict(self, X):
        N = len(X)
        y_hat = np.zeros(N)

        for i in range(N):
            dists = self.norm_func(self.X, X[i], self.norm_pars)
            idx = dists.argsort()[:self.K]
            gamma = self.weight_function(dists[idx], self.weight_pars)
            if self.mode:
                    y_hat[i] = gamma.dot(self.Y[idx]) /gamma.sum()
            else:
                    y_hat[i] = np.bincount(self.Y[idx], weights = gamma).argmax()
        return y_hat
