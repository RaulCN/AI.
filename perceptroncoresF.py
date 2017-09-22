# encoding: utf-8

'''
	Implementação da rede Perceptron
	Autor: Marcos Castro
'''

import random, copy

class Perceptron:

	def __init__(self, amostras, saidas, taxa_aprendizado=0.1, epocas=1000, limiar=-1):

		self.amostras = amostras # todas as amostras
		self.saidas = saidas # saídas respectivas de cada amostra
		self.taxa_aprendizado = taxa_aprendizado # taxa de aprendizado (entre 0 e 1)
		self.epocas = epocas # número de épocas
		self.limiar = limiar # limiar
		self.num_amostras = len(amostras) # quantidade de amostras
		self.num_amostra = len(amostras[0]) # quantidade de elementos por cada amostra
		self.pesos = [] # vetor dos pesos


	# função utilizada para treinar a rede
	def treinar(self):

		# adiciona -1 para cada amostra
		for amostra in self.amostras:
			amostra.insert(0, -1)

		# inicia o vetor de pesos com valores aleatórios pequenos
		for i in range(self.num_amostra):
			self.pesos.append(random.random())

		# insere o limiar no vetor de pesos
		self.pesos.insert(0, self.limiar)

		num_epocas = 0 # inicia o contador de épocas
		
		while True:

			erro = False # erro inicialmente inexiste

			# para todas as amostras de treinamento
			for i in range(self.num_amostras):
				
				u = 0
				'''
					realiza o somatório, o limite (self.num_amostra + 1) 
					é porque foi inserido o -1 em cada amostra
				'''
				for j in range(self.num_amostra + 1):
					u += self.pesos[j] * self.amostras[i][j]

				# obtém a saída da rede utilizando a função de ativação
				y = self.sinal(u)

				# verifica se a saída da rede é diferente da saída desejada
				if y != self.saidas[i]:

					# calcula o erro: subtração entre a saída desejada e a saída da rede
					erro_aux = self.saidas[i] - y

					# faz o ajuste dos pesos para cada elemento da amostra
					for j in range (self.num_amostra + 1):
						self.pesos[j] = self.pesos[j] + self.taxa_aprendizado * erro_aux * self.amostras[i][j]

					erro = True # se entrou, é porque o erro ainda existe
			
			num_epocas += 1 # incrementa o número de épocas

			# critério de parada é pelo número de épocas ou se não existir erro
			if num_epocas > self.epocas or not erro:
				break


	# função utilizada para testar a rede
	# recebe uma amostra a ser classificada e os nomes das classes
	# utiliza função sinal, se é -1 então é classe1, senão é classe2
	def testar(self, amostra, classe1, classe2):

		# insere o -1
		amostra.insert(0, -1)

		'''
			utiliza-se o vetor de pesos ajustado
			durante o treinamento da rede
		'''
		u = 0
		for i in range(self.num_amostra + 1):
			u += self.pesos[i] * amostra[i]

		# calcula a saída da rede
		y = self.sinal(u)

		# verifica a qual classe pertence
		if y == -1:
			print('A amostra pertence a classe %s' % classe1)
		else:
			print('A amostra pertence a classe %s' % classe2)


	def degrau(self, u):
		return 1 if u >= 0 else 0


	def sinal(self, u): # é a mesma função degrau bipolar
		return 1 if u >= 0 else -1


if __name__ == "__main__":


	print('\nA ou B?\n')

	# todas as amostras (total de 4 amostras)
	amostras = [[0.1, 0.4, 0.7], [0.3, 0.7, 0.2], 
				[0.6, 0.9, 0.8], [0.5, 0.7, 0.1]]

	# saídas desejadas de cada amostra
	saidas = [1, -1, -1, 1]

	# conjunto de amostras de testes
	testes = copy.deepcopy(amostras)

	# cria uma rede Perceptron
	rede = Perceptron(amostras=amostras, saidas=saidas, 
						taxa_aprendizado=0.1, epocas=1000)

	# treina a rede
	rede.treinar()

	for teste in testes:
		rede.testar(teste, 'A', 'B')


	'''
		Outro caso de teste: classificando cores
	'''

	print('\nAzul ou Vermelho?\n')

	amostras2 = [ [0.72, 0.82], [0.91, -0.69],
				[0.46, 0.80],   [0.03, 0.93],
				[0.12, 0.25],   [0.96, 0.47],
				[0.8, -0.75],   [0.46, 0.98],
				[0.66, 0.24],   [0.72, -0.15],
				[0.35, 0.01],   [-0.16, 0.84],
				[-0.04, 0.68],  [-0.11, 0.1],
				[0.31, -0.96],   [0.0, -0.26],
				[-0.43, -0.65],  [0.57, -0.97],
				[-0.47, -0.03],  [-0.72, -0.64],
				[-0.57, 0.15],   [-0.25, -0.43],
				[0.47, -0.88],   [-0.12, -0.9],
				[-0.58, 0.62],   [-0.48, 0.05],
				[-0.79, -0.92],  [-0.42, -0.09],
				[-0.76, 0.65],   [-0.77, -0.76]]

	saidas2 = [-1,-1,-1,-1,-1,-1, -1,-1,
				-1,-1,-1,-1,-1,1,1,1,1,
				1,1,1,1,1,1,1,1,1,1,1,1,1]

	testes2 = copy.deepcopy(amostras2)

	rede2 = Perceptron(amostras=amostras2, saidas=saidas2,
						taxa_aprendizado=0.1, epocas=1000)

	rede2.treinar()

	for teste in testes2:
		rede2.testar(teste, 'Vermelho', 'Azul')
