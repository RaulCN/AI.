import numpy as np


class MLP(object):
    
    def __init__(self, init_r1=0.001, init_r2=0.001, init_hidden_nodes=10, init_max_iter=100000):
        self._r1 = init_r1
        self._r2 = init_r2
        self._hidden_nodes = init_hidden_nodes
        self._max_iter = init_max_iter
        
        self._w = None
        self._theta = None
        self._v = None
        self._gamma = None
    
    def train(self, x, y, init_v=None, init_w=None, init_theta=None, init_gamma=None):
        self._w = init_w or np.random.randn(y.shape[0], self._hidden_nodes)
        self._theta = init_theta or np.random.randn(y.shape[0], 1)
        
        self._v = init_v or np.random.randn(self._hidden_nodes, x.shape[0])
        self._gamma = init_gamma or np.random.randn(self._hidden_nodes, 1)
        
        for n in range(self._max_iter):
            if n % 1000 == 0:
                print('Training at {}'.format(n))
            y_hat = self.predict(x)
            g = y_hat * (1 - y_hat) * (y - y_hat)
            b = logit(self._v @ x + self._gamma)
            e = b * (1 - b) * (self._w.T @ g)
            
            self._w += self._r1 * (g @ b.T)
            self._v += self._r2 * (e @ x.T)
            self._theta += (self._r1 * g).sum(axis=1).reshape((2, 1))
            self._gamma += (self._r2 * e).sum(axis=1).reshape((10, 1))
        
        return y - self.predict(x)

    def predict(self, x):
        return logit(self._w @ logit(self._v @ x + self._gamma) + self._theta)
