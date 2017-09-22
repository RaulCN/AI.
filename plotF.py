import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 3, 20)
y = np.linspace(0, 9, 20)
plt.plot(x, y)       # plotagem da linha
plt.plot(x, y, 'o')  # plotagem dos pontos
plt.show()           # <-- mostra o gráfico
