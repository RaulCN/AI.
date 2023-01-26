import numpy as np

class Perceptron:
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):

      self.w_ = np.zeros(1 + X.shape[1])
      self.errors_ = []

      for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
      return self

    def predict(self, X):
       
        phi = np.where(self.net_input(X) >= 0.0, 1, -1)
        return phi

    def net_input(self, X):
        # z = w · x + theta
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return z

    def test(self, X, y):
     
        errors = sum(1 for Xi, label in zip(X, y)
                    if self.predict(Xi) != label)
        return 100 * (errors / len(X))
    
    # TEST

ppn = Perceptron(eta=1, n_iter=2) 


X = [
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1],
    [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0],
    [1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
]

y = [1, -1, 1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1]


ppn.fit(np.array(X), y)
print(ppn.predict([1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]))
print(ppn.test(np.array(X), y))

# rodar em  python3
