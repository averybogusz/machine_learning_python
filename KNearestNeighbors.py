import numpy as np
import matplotlib.pyplot as plt

def uniform_weights(dists, epsilon = 1e-2): #
    K = len(dists)
    return np.array([1/K]*K)

def reciprocal_weights(dists, epsilon = 1e-2):
    return 1/(np.sqrt(dists) + epsilon)

def gauss_weights(dists, epsilon):
        ed = np.exp(dists)
        return ed/np.sum(ed)

def accuracy(y, y_hat):
    return np.mean(y == y_hat)

def R2(y, y_hat):
    return 1 - (np.sum(y-y_hat)**2/np.sum(y-y.mean())**2)
class KNN():
    def __init__(self, K, P = 2, weight_function = gauss_weights, epsilon = 1e-2, mode = 0): # this tells you when you define class it needs initial attributes
        self.mode = mode
        self.K = K
        self.P = P
        self.weight_function = weight_function
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X = X
        self.Y = y

    def  predict(self, X):
        N = len(X)
        y_hat = np.zeros(N)

        for i in range(N):
            dists = np.sum((self.X - X[i]) ** 2, axis = 1)
            idx = dists.argsort()[:self.K]
            gamma = self.weight_function(dists[idx], epsilon = self.epsilon)
            if self.mode:
                    y_hat[i] = gamma.dot(self.Y[idx]) /gamma.sum()
            else:
                    y_hat[i] = np.bincount(self.Y[idx], weights = gamma).argmax()
        return y_hat

def main_class():
        D = 2
        K = 3
        N = int(K * 1e3)

        X0 = np.random.randn((N // K), D) + np.array([2, 2])
        X1 = np.random.randn((N // K), D) + np.array([0, -2])
        X2 = np.random.randn((N // K), D) + np.array([-2, 2])
        X = np.vstack((X0, X1, X2))
        y = np.array([0] * (N // K) + [1] * (N // K) + [2] * (N // K))

        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y, alpha=.5)
        knn = KNN(9)
        knn.fit(X,y)
        y_hat = knn.predict(X)
        print(f"training Accuracy: {accuracy(y, y_hat):0.4f}")

def main_reg():
    N=200

    X = np.linspace(0,20,N).reshape(N,1)
    y = 3.5847 + 2.9574 * np.sqrt(X) + 4.9574*np.exp(-(X-1))- 6.948*np.exp(-(X - 7.47)**2) + np.random.randn(N,1)*.7
    plt.figure()
    plt.scatter(X,y)
    plt.show()

    knn = KNN(K=9, mode = 1)
    knn.fit(X,y)
    y_hat = knn.predict(X)
    print(f"Training R-Square: {R2(y,y_hat):0.4f}")
    plt.figure()
    plt.scatter(X,y)
    plt.plot(X,y_hat, color = '#000000', linewidth = 2)

if __name__ == "__main__":
    #main_class()
    main_reg()