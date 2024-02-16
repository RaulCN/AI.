import numpy as np
import pandas as pd

# O objetivo deste código é implementar um Perceptron, um modelo de classificação linear simples,
# e usá-lo para classificar amostras do conjunto de dados Iris em duas classes.

class Perceptron(object):
    """
    Parâmetros:
    -----------
    eta: taxa de aprendizagem
    n_iter: número de passagens (épocas)
    
    Atributos
    ----------
    w_ : pesos após o ajuste
    errors_ : número de classificações incorretas em cada época
    """

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Ajusta os dados de treinamento.

        Parâmetros
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Vetores de treinamento, onde n_samples é o número de amostras e
            n_features é o número de características.
        y : array-like, shape = [n_samples]
            Valores alvo.

        Retorna
        -------
        self : object
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
            print(f"Época {i+1}: erros = {errors}")
            print(f"Pesos atualizados: {self.w_}")
        return self

    def net_input(self, X):
        """Calcula a entrada líquida."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Retorna o rótulo de classe após a etapa de unidade."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
y = df.iloc[0:100, 4].values
y = np.where( y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
