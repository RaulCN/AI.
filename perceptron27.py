#não está funcionando
#https://gist.githubusercontent.com/iamPres/ef9c1799e4ad99e473e20f18d73583fc/raw/065b38b3e64b482314349f997b0138d85b57eb99/Sigmoid%2520Neuron.py

import random
import math

class Train:	
	def __init__(self):
		self.train = [[0,0,1],[0,1,0],[1,1,0]]
		#Initialize training set
		self.weights = [5,5]
		self.bias = 1
		self.learningRate = 0.1
	#true if fired state == answers in training set
		self.test = [0,1]
		self.error = [1,1,1]
		self.sig = [0,0,0]
		self.output = [0,0,0]
		self.testOut = 0
		self.testSig = 0
	
	def trainIt(self):	
		#while one or more of the answers are wrong
		while self.error[0] > 0.1 or self.error[1] > 0.1 or self.error[0] < -0.1 or self.error[1] < -0.1:
			
			for i in range(3):
				#Perceptron formula
				self.output[i] = self.train[i][0]*self.weights[0]+self.train[i][1]*self.weights[1]+self.bias
				self.sig[i] = math.erf(self.output[i])
			
				self.error[i] = self.sig[i]-self.train[i][2]
				
				for j in range(2):
					
					self.weights[j] += -self.learningRate*(self.error[0]**2+self.error[1]**2+self.error[2]**2)*self.train[i][j]
					self.bias += -self.learningRate*(self.error[0]+self.error[1]+self.error[2])
				
				for j in range(2):
					print("W"+str(j)+": "+str(self.weights[j]))
				for j in range(3):
					print(self.sig[j])
				
	def Test(self):
		print("#----------------------------------#")
		#random point to test perceptron!
		self.testOut = self.test[0]*self.weights[0]+self.test[1]*self.weights[1]+self.bias 
		self.testSig = math.erf(self.testOut)
	
		print(self.testSig)
		for i in range(2):
			print(self.test[i])
		
		
			
#initialize object and call methods			
ptron = Train()
ptron.trainIt()
ptron.Test()
	
