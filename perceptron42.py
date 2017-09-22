
#Funcionando só n sei se ta fazendo oq é pra fazer.
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

theta   = 0.0
weights = [0.0, 0.0, 0.0]
eta     = 1.0

E = [1, 0, 0], 1
D = [0, 1, 1], 0
H = [1, 1, 1], 1

def update(avec, target):
    global theta
    global weights
    global eta

    print ("avec = %s" % avec)
    print ("weights = %s" % weights)

    inr = sum(map(lambda a,b: a*b, avec, weights))
    print ("inr = %0.3f" % inr)

    ar  = 1 if inr >= theta else 0
    print ("ar = %i" % ar)

    if ar != target:
        for i in xrange(len(weights)):
            weights[i] = weights[i] + (eta * (target - ar) * avec[i])
        theta = theta - (eta * (target - ar))

    print ("theta = %0.3f" % theta)
    print ("weights = %s" % weights)
    print
    print

update(*E)
update(*D)
update(*H)
