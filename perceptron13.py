#!/lusr/bin/python
#não funciona
#https://gist.githubusercontent.com/stober/2938732/raw/147e07e9ae9554f60b3836e806a22f21183680a5/perceptron.py
"""
Author: Jeremy M. Stober
Program: PERCEPTRON.PY
Date: Friday, April  4 2008
Description: A simple Perceptron implementation.
"""

import os, sys, getopt, pdb
from numpy import *
from numpy.random import *
import pylab

class Perceptron(object):

    def __init__(self, size):
        self.weights = zeros(size + 1) # include bias term

    def train(self, input, response):
        x = append(input,1) # include bias term
        y = dot(self.weights, x)

        # For now we'll just worry about the sign of the response.
        error = abs(sign(response) - sign(y))
        if error > 0:
            self.weights = self.weights + sign(response) * x

        # Return the networks output.
        return sign(y)

    def test(self, input):
        x = append(input, 1) # include bias term

        # The output of a trained perceptron - no learning.
        return sign(dot(self.weights,x))

def test(name):

    if name == 'and':

        perceptron = Perceptron(2)

        def label(input):
            if all(input):
                return 1
            else:
                return -1

        for i in range(100):
            input = randint(0,2,2)
            print (input, label(input), perceptron.train(input, label(input))

        print perceptron.test(array([1,1]))
        print perceptron.test(array([0,1]))
        print perceptron.test(array([1,0]))
        print perceptron.test(array([1,0]))

    if name == 'planar':

        perceptron = Perceptron(2)

        # We use a 2-d normal to generate a random hyperplane.
        weights = array([0.0,1.0,0.0])
        def label(input):
            x = append(input,1.0)
            if dot(weights,x) > 0:
                return 1
            else:
                return -1

        input = normal(size = (100,2))

        for i in range(len(input)):
            point = input[i]
            print point, label(point), perceptron.train(point, label(point))

        red = []
        green = []

        for i in range(len(input)):
            point = input[i]
            if perceptron.test(point) > 0:
                red.append(point)
            else:
                green.append(point)

        print perceptron.weights
        pylab.plot([r[0] for r in red], [r[1] for r in red] ,'.', color = 'red')
        p = pylab.plot([g[0] for g in green], [g[1] for g in green] ,'.', color = 'green')
        pylab.plot([-3,3],[0,0])
        pylab.axis([-3,3,-3,3])
        pylab.show()

def plotline(weights):
    axis = pylab.axis() #[xlo,xhi,ylo,yhi]

    # TODO: finish function for easily plotting lines

def main():

    def usage():
	print sys.argv[0] + "[-h] [-d]"

    try:
        (options, args) = getopt.getopt(sys.argv[1:], 'dh', ['help','debug'])
    except getopt.GetoptError:
        # print help information and exit:
        usage()
        sys.exit(2)

    for o, a in options:
        if o in ('-h', '--help'):
            usage()
            sys.exit()
	elif o in ('-d', '--debug'):
	    pdb.set_trace()

    test('planar')

if __name__ == "__main__":    main()
