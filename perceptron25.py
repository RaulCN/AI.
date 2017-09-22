#Não está funcionando
#Fonte https://gist.githubusercontent.com/iamPres/91bf1ca2c640b9fccfc71c2ec301c962/raw/fac517634c7bf3d3654c13b382bd96ab53ad8584/neuralNet-XOR.py

import random #used for learning later

class norGate:
	#Class for nor gate perceptron
	def __init__(self):
		self.weights = [-20,-20]
		self.bias = 30 
		self.fired = False
		
	def fire(self):
		#if the inputs multiplied by the weights plus the bias is > zero, fire
		if net.input[0][0]*self.weights[0]+net.input[0][1]*self.weights[0]+self.bias <= net.threshold:
			self.fired = False
		else:
			self.fired = True
					
class orGate:
	#Class for the or gate perceptron
	def __init__(self):
		self.weights = [20,20]
		self.bias = -10
				
	def fire(self):
		if net.input[0][0]*self.weights[0]+net.input[0][1]*self.weights[0]+self.bias <= net.threshold:
			self.fired = False
		else:
			self.fired = True
		
class andGate:
	#Class for the and-gate perceptron
	#Output of the neural network
	def __init__(self):
		self.weights = [20,20]
		self.bias = -30
		
	def fire(self):
		#if the outputs from the hidden layer perceptrons multiplied by the weights plus the bias is > zero, output a 1
		if int(orG.fired)*self.weights[0]+int(norG.fired)*self.weights[0]+self.bias <= net.threshold:
			self.fired = False
		else:
			self.fired = True
		
class neuralNet:
		#Class for defining the neural network's global variables
	def __init__(self):
		#an array for learning later
		self.input = [[int(input()),int(input()),False],[1,1,False],[1,0,True],[0,1,True]]	
		self.learningRate = 0.1 #not used yet
		self.threshold = 0 
		
#-----------------------------------------------#
		
net = neuralNet()
andG = andGate()
norG = norGate()
orG = orGate()

orG.fire()
norG.fire()
andG.fire()

print(norG.fired)
print(orG.fired)
print(andG.fired)


