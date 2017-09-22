import numpy as np
#não sei oq faz

class Perceptron(object):
    def __init__(self, learning_rate=0.01, iterations=10):
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, X, y):

        self.weights = np.zeros(1 + X.shape[1])

        for _ in range(self.iterations):
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[0] += update
                self.weights[1:] += update * xi
        return self

    def net_input(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
