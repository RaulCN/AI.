import numpy as np
import pandas as pd


# The principle is that it will take each sample and iteratively use the activation function on the dot product of the
# weights and the input. The number of iterations is defined by the epoch

class Perceptron(object):
    """
    Parameters:
    -----------
    eta: learning rate
    n_iter: number of passes (epoch)
    Attributes
    ----------
    w_ : weights after fitting
    errors_ number of misclassifications in every epoch
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):

        """fit training data
        X:  [n_samples, n_features]
            training vector
        y:  [n_samples]
            target
        """

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
y = df.iloc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
