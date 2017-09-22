#não funciona
#fonte https://gist.github.com/zenathark/50266ceb73dbec722c54

"""
.. moudle:: perceptron
   :platform: Unix, Windows
   :synopsis: Perceptron simulator.
      This module creates a perceptron and trains it to learn the AND
      operation.
      This particular perceptron uses a linear error as learning method and
      a simple activation function defined by

    \[
      f(x) =
        \begin{cases}
          0 & \text{if } x > K \\
          1 & \text{otherwise}
        \end{cases}
    \]

    where K is defined a priori empirically. For the AND problem, the known K
    value that works is 0.5.

.. moduleauthor:: Juan Carlos Galan Hernandez <juan.galanhz@udlap.mx>

"""
from __future__ import division
import scipy as np
import itertools as itr


inputs = np.array(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])

output = np.array(
    [
        0,
        0,
        0,
        1
    ])

#Activation function
f = lambda x, K: 1 if x > K else 0

#error function
#because the activation function is an horizontal line, backpropagation cannot
#be used. Instead, we will use a linear error and move along the function
#in direction of the error
E = lambda y, y_est: (y - y_est)

#weights
w = np.random.rand(2)

#K value (knowning a priori that it works)
K = 0.5

#Learning rate (above 0.9ish does not work)
alpha = 0.1

#Stoping criteria
stop_error = 0.005
#The error is the accumulative of each input-output pair
accum_error = 1
while accum_error > stop_error:
    accum_error = 0
    for x, y in itr.izip(inputs, output):
        #activate perceptron
        y_est = f(float(np.dot(w, x)), K)
        #learning
        error = E(y, y_est)
        #update weights
        w = w + alpha * error * x
        #Accumulate error for stopping criteria
        accum_error += abs(error)
    #Calculate the mean of the errors (stopping criteria)
    accum_error = 1 / len(inputs) * accum_error
    print("Accum error {}".format(accum_error))
#testing the trained network
print("Testing ANN")
for i in inputs:
    y_est = f(float(np.dot(w, i)), K)
    print("input {} output {}".format(i, y_est))
