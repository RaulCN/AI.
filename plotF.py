#passos simples para criar um gráfico no matplotlib
import matplotlib.pyplot as plt #importando a biblioteca gráfica
import numpy as np #importando a biblioteca para operações matemáticas

x = np.linspace(0, 3, 20) #add os pontos nas das coordenadas
y = np.linspace(0, 9, 20) #add os pontos nas absissas
plt.plot(x, y, 'b')       # plotagem da linha
plt.plot(x, y, 'o')  # plotagem dos pontos #  https://goo.gl/UqyxhC
plt.show()           # <-- mostra o gráfico
