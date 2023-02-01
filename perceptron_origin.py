#!/usr/bin/env python   
# -*- coding: utf-8 -*-
#original https://gist.github.com/Linusp/348b20c06d856bc6271b#file-perceptron_origin-py

'''
Esse código é uma implementação do algoritmo de Perceptron em Python do link acima 
O Perceptron é um algoritmo de aprendizado de máquina que pode ser usado para classificar dados em dois grupos.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

'''
O código importa três bibliotecas: numpy, pyplot e animation, que são usadas para cálculos matemáticos, plotagem de gráficos e animação, respectivamente. 
Também é importada a biblioteca time, que será usada para gerar intervalos de tempo na animação.'''

#Inicializa a figura que será usada para desenhar o gráfico
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

#Definindo um array de números com 3 colunas
x = np.array([[1, 2, 1], [2, 4, -1], [3, 6, 1], [4, 8, -1]]) #retirei o arquivo txt do código original

weight = np.array([0.0, 0.0])

'''
"bias" é uma variável que representa um deslocamento ou offset na equação de classificação usada na função "is_error_classify". 
Ele é adicionado ao resultado da multiplicação dos pesos e dos valores de entrada. Se o valor do "bias" é 0, então ele não terá nenhum impacto no resultado final. 
No código que estamos analisando, o valor inicial de "bias" é definido como 0.0, o que significa que não haverá nenhum deslocamento no resultado da equação de classificação.
'''

bias = 0.0
#Define o fator de aprendizagem
fai = 1.0

def is_error_classify(arr):
    result = (np.dot(arr[0:2], weight) + bias) * arr[2]
    return result <= 0

'''
Essa função verifica se houve erro na classificação dos dados.
Ela pega os primeiros dois valores de "arr" (que representam as características dos dados),
multiplica pelo peso "weight" e soma o bias. O resultado é multiplicado pelo terceiro valor de "arr"
(que representa a classe dos dados).
Por fim, a função retorna "True" se o resultado for menor ou igual a zero, ou "False" caso contrário.
'''

def first_error():
    global x
    for i in range(0, len(x)):
        if is_error_classify(x[i]):
            return i
    return -1

'''
Essa função procura por um erro na classificação dos dados.
Ela utiliza uma variável global x que é uma matriz com as informações dos dados.
A função faz um loop através dos dados em x usando o range(0, len(x)).
Em cada iteração, a função verifica se houve erro na classificação dos dados chamando a função is_error_classify().
Se houve erro, a função retorna o índice do elemento que causou o erro.
Caso não haja erro, a função retorna -1.
'''



def update(i):
    global weight, bias, x
    weight = weight + x[i][0:2] * x[i][2] * fai
    bias = bias + x[i][2] * fai
    print ("Error points(%d): (%f %f);Weight: (%f %f - %d); bias: %f" %(i,
    x[i][0], x[i][1], x[i][2], weight[0], weight[1], bias))

'''
A função update é responsável por atualizar os valores de peso e viés a partir do resultado do erro classificado na função is_error_classify.

A variável i representa o índice do ponto de erro na matriz x. 
O peso é atualizado somando o valor do ponto de erro na primeira e segunda posição multiplicado pelo valor da classe na terceira posição multiplicado pela taxa de aprendizado fai.

O viés é atualizado somando o valor da classe multiplicado pela taxa de aprendizado. Além disso, as informações são exibidas na tela após a atualização.
'''

def plot_all_points():
    global x, ax1
    for point in x:
      if point[2] == 1:
            ax1.plot(point[0], point[1], 'ro') #'ro' é uma representação gráfica de ponto vermelho. No código acima, se a terceira posição de "point" for igual a 1, então um ponto vermelho será plotado no gráfico usando "ax1.plot(point[0], point[1], 'ro')". Caso contrário, se a terceira posição de "point" for igual a -1, um ponto azul será plotado usando "ax1.plot(point[0], point[1], 'bo')".
      else:
            ax1.plot(point[0], point[1], 'bo') #'bo' representa "blue circles". É usado para plotar pontos com valor negativo (-1) na tela. A função ax1.plot(point[0], point[1], 'bo') é usada para desenhar círculos azuis (representando valores negativos) no eixo x e y, usando as coordenadas point[0] e point[1], respectivamente.

'''
A função plot_all_points é usada para plotar todos os pontos de dados armazenados no array x no gráfico. 
Para cada ponto no array x, é verificado se o último elemento do ponto é igual a 1. Se for verdadeiro, então o ponto é plotado como um ponto vermelho ('ro') no gráfico. 
Se não, o ponto é plotado como um ponto azul ('bo') no gráfico. A função usa o objeto ax1 que representa o gráfico para desenhar os pontos
'''


# função principal da animação, atualiza a exibição da linha de separação a cada interação
def animate(i):
    global weight, bias, x # variáveis globais
    
    err_index = first_error() # procura por um erro de classificação
    if err_index >= 0: # se houver erro, atualiza os pesos e bias
        update(err_index)
    else: # se não houver erro, encerra o programa
        exit()
        
    ax1.clear() # limpa o gráfico
    plot_all_points() # plota todos os pontos
    
    # cria uma malha para o gráfico
    x_range = np.arange(-5, 15, 0.025)
    y_range = np.arange(-5, 15, 0.025)
    X, Y = np.meshgrid(x_range, y_range)
    
    # calcula a função da linha de separação
    f = weight[0] * X + weight[1] * Y + bias
    
    # adiciona a linha de separação ao gráfico
    ax1.contour(X, Y, f, [0], colors=('green'))
    
    # define o limite do eixo x e y
    plt.xlim([-5, 15])
    plt.ylim([-5, 15])

# início do programa
if __name__ == '__main__':
    plot_all_points() # plota todos os pontos
    
    # cria a animação com a função animate e intervalo de 1 segundo
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    
    # exibe o gráfico
    plt.show()
