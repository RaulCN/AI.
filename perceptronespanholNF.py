#pra que ele serve?
import numpy as np

class Perceptron:
    """Clasificador Perceptron baseado na descrição do livro
    "Python Machine Learning" de Sebastian Raschka. (qual página?)
    Parametros
    fonte desse código:
    https://gist.github.com/Edux87/a6793ce1041a62ae8a08c761a9478cd5
    ----------
    eta: float
        Taxa de aprendizagem.
    n_iter: int
        Pasadas sobre el dataset.
    Atributos
    ---------
    w_: array-1d
        Pesos actualizados depois do ajuste
    errors_: list
        Quantidade de erros de classificação em cada pasada
    """
    def __init__(self, eta=0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Ajustar dados de treinamento
        Parâmetros
        ----------
        X:  array like, forma = [n_samples, n_features]
            Vetores de entrenamiento onde n_samples é o número de muestras e
            n_features é o número de carácteristicas de cada muestra.
        y:  array-like, forma = [n_samples].
            Valores de destino
        n_samples = muestras (filas)
        n_features = caracteristicas (columnas)
        Returns
        -------
        self:   object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def predict(self, X):
        """Devolver clase usando função escalón de Heaviside.
        phi(z) = 1 si z >= theta; -1 em outro caso
        """
        phi = np.where(self.net_input(X) >= 0.0, 1, -1)
        return phi

    def net_input(self, X):
        """Calcular o valor z (net input)"""
        # z = w · x + theta
        z = np.dot(X, self.w_[1:]) + self.w_[0]
        return z

    def test(self, X, y):
        """Test classifier on samples, and returns error/total percentage."""
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
