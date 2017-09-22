#Inspiração https://gist.github.com/borgwang/25cc5bc12c26c3f57755a78f9b868b7b
#Está funcionando

import numpy as np
import matplotlib.pyplot as plt

# positivo: (3, 3), (4, 3)
# negativo: (1, 1), (0, 0)
X = np.array([(3, 3), (4, 3), (1, 1), [0, 0]])
Y = np.array([1, 1, -1, -1])

w = np.zeros(2).T
b = 0
leanring_rate = 0.01

for _ in range(30): 
    d_w = np.zeros(2).T
    d_b = 0
    for x, y in zip(X, Y):
        fx = np.dot(x, w) + b
        loss = -y * fx
        if loss >= 0:
            d_w += -y * x.T
            d_b += -y
    # update w, b
    w = w - leanring_rate * d_w
    b = b - leanring_rate * d_b

print(w, b)
# plot data
plt.scatter(X[:2][:, 0], X[:2][:, 1], c='b')
plt.scatter(X[2:][:, 0], X[2:][:, 1], c='r')
# plane
plane = np.array([(0, 5), (2.5, 0)])
plt.plot(plane[:, 0], plane[:, 1], color='black')
plt.show()
