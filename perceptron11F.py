"""código original http://codegist.net/snippet/python/perceptronpy_lite-david_python
código está funcionando"""

import numpy as np 
import random

class Perceptron:

	def __init__(self):
		self.weights = np.random.rand(1,3)

	def step(self,x):
		return np.piecewise(x,[x<0,x>=0],[0,1])

	def weightedSum(self,x):
		return np.dot(x,np.transpose(self.weights))

	def train(self,X,Y,iterations):
		X = np.hstack((np.ones((X.shape[0],1)),X))
		#for i in range(iterations):
		i=0
		while not np.array_equal(np.reshape(self.predict(X),(1,4))[0],Y):
			i+=1
			rint = random.randint(0,3) 
			x = X[rint]
			d = Y[rint]
			v = self.weightedSum(x)
			y = self.step(v)
			self.weights = self.weights + (d-y)*x
		print ("Número de iterações para convergir"), i

	def predict(self,x):
		if(x.shape[1] != self.weights.shape[1]):
			x = np.hstack((np.ones((x.shape[0],1)),x))
		v = self.weightedSum(x)
		return self.step(v)

if __name__ =='__main__':
	p = Perceptron()
	X = np.array([[0,0],[0,1],[1,0],[1,1]])
	Y = np.array([0,0,0,1])
	p.train(X,Y,100)
	print (p.predict(X))
