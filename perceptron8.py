import numpy as np

class Perceptron(object):
    """Perceptron classifier (classificador binário).
    
    Parâmetros
    ------------
    eta : float (padrão=0.01)
        Taxa de aprendizado (entre 0.0 e 1.0)
    n_iter : int (padrão=10)
        Número de passos (épocas) sobre o conjunto de treinamento

    Atributos
    -----------
    w_ : array 1D
        Pesos após o treinamento
    errors_ : list
        Número de classificações incorretas em cada época
    """
    
    def __init__(self, eta=0.01, n_iter=10):
        # Inicializa hiperparâmetros
        self.eta = eta          # Taxa de aprendizado: controla o tamanho do passo nas atualizações
        self.n_iter = n_iter    # Número de épocas: quantas vezes o modelo verá os dados

    def fit(self, X, y):
        """Treina o modelo com os dados.
        
        Parâmetros
        ----------
        X : array-like, shape = [n_amostras, n_caracteristicas]
            Dados de treinamento
        y : array-like, shape = [n_amostras]
            Valores alvo (rótulos)

        Retorna
        -------
        self : object
        """
        
        # Verificação de formato dos dados
        if len(X.shape) != 2:
            raise ValueError("X deve ser uma matriz 2D (n_amostras x n_caracteristicas)")
        if len(y.shape) != 1:
            raise ValueError("y deve ser um vetor 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem ter o mesmo número de amostras")

        # Inicialização dos pesos:
        # - +1 para o termo de bias (w_[0])
        # - Pesos iniciais = 0 (pode ser alterado para outros valores)
        self.w_ = np.zeros(1 + X.shape[1])  # [bias, w1, w2, ..., wn]
        self.errors_ = []  # Lista para armazenar erros por época

        # Loop de treinamento por épocas
        for _ in range(self.n_iter):
            errors = 0  # Contador de erros na época atual
            
            # Para cada amostra (xi) e rótulo (target) no conjunto de dados
            for xi, target in zip(X, y):
                
                # Reformata xi para ser uma matriz 2D (1 amostra, n características)
                xi_2d = xi.reshape(1, -1)
                
                # Calcula a atualização:
                # (target - prediction) = 0 se acertou, ±2 se errou
                update = self.eta * (target - self.predict(xi_2d))
                
                # Atualiza os pesos:
                # w_[1:] = pesos das características (sem bias)
                # w_[0] = bias (intercepto)
                self.w_[1:] += update * xi  # Atualização proporcional às características
                self.w_[0] += update       # Atualização do bias
                
                errors += int(update != 0.0)  # Conta 1 erro se update ≠ 0
                
            self.errors_.append(errors)  # Armazena erros da época
            
        return self

    def net_input(self, X):
        """Calcula a entrada líquida (combinação linear dos pesos e características)"""
        if len(X.shape) != 2:
            raise ValueError("X deve ser uma matriz 2D para cálculo da entrada líquida")
        return np.dot(X, self.w_[1:]) + self.w_[0]  # X*w + bias

    def predict(self, X):
        """Retorna a classe prevista após função degrau"""
        if len(X.shape) != 2:
            raise ValueError("X deve ser uma matriz 2D para previsão")
        # Função degrau: retorna 1 se entrada líquida ≥ 0, -1 caso contrário
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# =================================================================
# Exemplo de Uso com Dados de Entrada e Explicação dos Resultados
# =================================================================
if __name__ == "__main__":
    # Dados de Treinamento Artificialmente Criados (Linearmente Separáveis)
    # Características (X): [Feature1, Feature2]
    # Rótulos (y): -1 (Classe A) ou 1 (Classe B)
    X = np.array([
        [2, 3],  # Classe B
        [3, 4],  # Classe B
        [1, 1],  # Classe A
        [0.5, 0.5]  # Classe A
    ])
    y = np.array([1, 1, -1, -1])

    print("\nDados de Entrada:")
    print("Características (X):\n", X)
    print("Rótulos (y):", y)

    # Cria e Treina o Perceptron
    ppn = Perceptron(eta=0.1, n_iter=5)  # eta maior para convergência mais rápida
    ppn.fit(X, y)

    # Resultados Esperados e Explicação
    print("\n=== Resultados do Treinamento ===")
    print("Pesos Iniciais (w): [bias, w1, w2] = [0, 0, 0]")
    print("Pesos Finais:", ppn.w_)
    print("Erros por Época:", ppn.errors_)
    
    # Fazendo Previsões
    print("\nPrevisões para os Dados de Treinamento:")
    print("Previsto:", ppn.predict(X))
    print("Real:    ", y)

    # Explicação Didática
    print("\n*** Explicação dos Resultados ***")
    print("1. Erros por Época devem diminuir a cada iteração até zerar:")
    print("   - Isso indica que o modelo está aprendendo")
    print("   - Dados são linearmente separáveis, então convergência é esperada")
    
    print("\n2. Pesos Finais definem a fronteira de decisão:")
    print(f"   - Equação da reta: {ppn.w_[1]:.2f}x1 + {ppn.w_[2]:.2f}x2 + {ppn.w_[0]:.2f} = 0")
    print("   - Pontos acima desta reta são classificados como 1")
    print("   - Pontos abaixo como -1")

    print("\n3. Previsões devem coincidir com os rótulos reais:")
    print("   - Indica que o modelo generalizou bem para os dados de treinamento")
