#https://gist.github.com/reetawwsum/a51d3a41fc066f37fcb8
#Corrigi erro dos parênteses e troquei x range para range
from random import choice
from numpy import array, dot, random
import matplotlib.pyplot as p
import pylab
from pylab import plot, ylim

class Perceptron:
	'Perceptron Class for OR and AND gate'

	weights = random.rand(3)
	errors = []

	def __init__(self, training_data):
		self.training_data = training_data

	def learn(self, n):
		for i in range(n):
			x, expected = choice(self.training_data)
			result = dot(Perceptron.weights, x)
			error = expected - self.threashold(result)
			Perceptron.errors.append(error)
			Perceptron.weights += error * x

	def display(self, text):
		self.text = text
		print (self.text)
		for x, _ in self.training_data: 
			result = dot(x, Perceptron.weights) 
			print (x[:2], self.threashold(result))

	def plot_graph(self):
		ylim([-1,1])
		p.plot(Perceptron.errors)
		pylab.show()

threashold = lambda self, x: 0 if x < 0 else 1
setattr(Perceptron, 'threashold', threashold)

if __name__ == '__main__':
	training_data_for_AND_perceptron = [ 
		(array([0,0,1]), 0), 
		(array([0,1,1]), 0), 
		(array([1,0,1]), 0), 
		(array([1,1,1]), 1) 
	]

	AND_perceptron = Perceptron(training_data_for_AND_perceptron)
	AND_perceptron.learn(100)
	AND_perceptron.display('Output of AND perceptron:')
	AND_perceptron.plot_graph()

	training_data_for_OR_perceptron = [ 
		(array([0,0,1]), 0), 
		(array([0,1,1]), 1), 
		(array([1,0,1]), 1), 
		(array([1,1,1]), 1)
	]

	OR_perceptron = Perceptron(training_data_for_OR_perceptron)
	OR_perceptron.learn(100)
	OR_perceptron.display('Output of OR perceptron:')
	OR_perceptron.plot_graph()
