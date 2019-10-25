import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

def accuracy(y,y_hat):
    return np.mean( y== y_hat)

class NaiveBayesClassifier():
    def fit(self, X, y, naive=0):
        self.likelihoods =dict()
        self.priors = dict()
        self.K = set(y)

        for k in self.K:
            X_k  = X[y==k]
            self.likelihoods[k] = {"mu" : X_k.mean(axis = 0), "SIGMA": (1 - naive) * np.cov(X_k.T) + naive * np.diag(np.diag(np.cov(X_k.T)))} #note the Naive indicator will simply tell python whether to grab the covariance matrix, or its diagonal
            self.priors[k] = len(X_k)/len(X)
        pass

    def predict(self, X): #not the same X as before, though it still needs D columns.

        P = np.zeros((len(X), len(self.K)))

        for k,l in self.likelihoods.items(): #k is the label, l is the parameters, going through unpacking dictionary
            P[:,k] = mvn.logpdf(X, l["mu"], l["SIGMA"]) + np.log(self.priors[k])# fixing k, we'll store these in the right columns. remember mu and sigma are vectors

        return P.argmax(axis=1)

        pass


ex_dict = {0: {"mu":70, "SIGMA":40}, 1: {"mu":63, "SIGMA":7}} # this is basically how the dictionary works, each number is a k im pretty sure
for k, l in ex_dict.items():
    print(k)
    print(l)

def main():
    D = 2
    K = 3
    N = int(K * 1e3)

    X0 = np.random.randn((N//K), D) + np.array([2,2])
    X1 = np.random.randn((N//K),D) + np.array([0,-2])
    X2 = np.random.randn((N//K),D) + np.array([-2,2])
    X = np.vstack((X0, X1, X2))
    y = np.array([0]*(N//K) + [1] *(N//K) + [2] * (N//K))

    plt.figure()
    plt.scatter(X[:,0], X[:,1], c = y, alpha = .5)

    nb = NaiveBayesClassifier()
    nb.fit(X,y, naive=0)
    y_hat = nb.predict(X)
    print(f"Training accuracy: {accuracy(y, y_hat):0.4f}")
    pass

if __name__ == "__main__":
    main()
