#Funcionando, mas n me parece um perceptron

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pts0 = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0),
]

pts1 = [
    (0, 0, 1),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]

fig = plt.figure()
axe = Axes3D(fig)

axe.scatter([pt[0] for pt in pts0], [pt[1] for pt in pts0], [pt[2] for pt in pts0], color='r', marker='o')
axe.scatter([pt[0] for pt in pts1], [pt[1] for pt in pts1], [pt[2] for pt in pts1], color='b', marker='^')
#axe.plot((1,1,1), 'r--')

plt.show()
