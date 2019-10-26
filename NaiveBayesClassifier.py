import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

'''
The generalized NaiveBayesClassifier assumes a user-supplied pdf, written as a function. This function should take a 
in data, a vector of means, and a variance-covariance matrix. I suppose it could also simply take data and the means in certain cases. 
Anything more complex, ie, if you want to define the shape of the distribution by some function of sigma and the means, rather than directly, 
will require changes within your pdf as the code isn't able to handle that. 
'''



class NaiveBayesClassifier():

    def fit(self, X, y, naive=0):
        self.likelihoods =dict()
        self.priors = dict()
        self.K = set(y)

        for k in self.K:
            X_k  = X[y==k]
            self.likelihoods[k] = {"mu" : X_k.mean(axis = 0), "SIGMA": (1 - naive) * np.cov(X_k.T) + naive * np.diag(np.diag(np.cov(X_k.T)))} #features independent of one another
            self.priors[k] = len(X_k)/len(X)
        pass

    def predict(self, X, user_pdf = mvn.pdf, include_sigma = 1): #not the same X as before, though it still needs D columns.

        # input_pdf allows the user to specify an arbitrary pdf that takes either some data and the mean of the distribution
        # or, alternatively, the data, the mean, and the variance-covariance matrix (variance in the univariate case).
        # I'm not really sure about generalizing further since you can define so many distributions by so many different parameters.
        def input_pdf(*args, user_pdf, include_sigma):
            X_data=args[0]
            mu_mat=args[1]
            sigma_mat=args[2]
            if include_sigma == 1:
                return user_pdf(*args)
            else:
                return user_pdf(args[0], args[1])

        P = np.zeros((len(X), len(self.K)))

        for k,l in self.likelihoods.items(): #k is the label, l is the parameters, going through unpacking dictionary
            P[:,k] = input_pdf(X, l["mu"], l["SIGMA"], user_pdf = user_pdf, include_sigma = include_sigma) * self.priors[k] # fixing k, we'll store these in the right columns. remember mu and sigma are vectors

        return P.argmax(axis=1)

        pass

