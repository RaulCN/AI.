#!/usr/bin/env python3
#não funciona, pq?

from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

normed = lambda v: v / np.linalg.norm(v)
# orthogonal vector in 2 space
orthogonal = lambda v: np.cross(v, [0, 0, 1])[:2]
pltvec = lambda v, **kwargs: plt.plot([0, v[0]], [0, v[1]], **kwargs)
classify = lambda D, weight: (np.dot(D, weight) > 0) * 2 - 1
score = lambda D, weight, y: np.count_nonzero(classify(D, weight) == y)

N = 200
# generate N random x,y points in [-1, 1] x [-1, 1]
D = np.random.ranf(size=(N, 2)) * 2 - 1
Dx, Dy = D.T

# generate random weight
weight = normed(np.random.ranf(size=2))
boundary = orthogonal(weight)

noise = .1
toflip = np.random.ranf(N) < noise
y = classify(D, weight)
y[toflip] = -1 * y[toflip]
Dplus, Dminus = D[y == 1], D[y == -1]

def train(D, Y, iters=3):
  nsamples, nfeatures = D.shape
  weight = np.zeros(nfeatures)
  trace = [weight]
  for i in range(iters):
    D_perm = D[np.random.shuffle(np.arange(nsamples))][0]
    for x, y in zip(D_perm, Y):
      activation = y * np.dot(x, weight)
      if activation <= 0:
        weight += y * x
        trace.append(np.copy(weight))
  return np.array(trace)

trace = train(D, y)
trace_norm = trace / np.linalg.norm(trace, axis=1).max()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
for i, (x, y) in enumerate(trace_norm):
  ax.plot3D([0, x], [0, y], i, color='b', alpha=0.5)

###
plt.figure()
plt.plot(Dplus[:, 0], Dplus[:, 1], 'b.')
plt.plot(Dminus[:, 0], Dminus[:, 1], 'r.')
pltvec(weight, color='g', label='weight')
pltvec(boundary, color='k', label='boundary')
pltvec(-boundary, color='k')
#for w in trace:
#  pltvec(w)
plt.legend()

plt.show()
