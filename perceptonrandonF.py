import random

class Train:	
	def __init__(self):
		self.train = [[0,0,1],[0,1,1],[1,1,0]]
		#Inicializar o conjunto de treinamento
		self.fired = [False,False,False] #ativar quando exceder 
		self.threshold = 0
		self.weights = [random.randrange(0,10),random.randrange(0,10)]
		self.bias = random.randrange(-1,1)
		self.learningRate = 0.5
	#true if fired state == answers in training set
		self.correct = [False,False,False] 
		self.test = [0,1]
		self.testFired = 0
		self.error = [1,1,1]
	
	def trainIt(self):	
		#while one or more of the answers are wrong
		while self.error[0] > 0.1 or self.error[0] < 0 or self.error[1] > 0.1 or self.error[1] < 0 or self.error[2] > 0.1 or self.error[2] < 0:
			
			for i in range(3):
				#Perceptron formula
				if self.train[i][0]*self.weights[0]+self.train[i][1]*self.weights[1]+self.bias < self.threshold:
					#under thresh, dont fire
					self.fired[i] = False
				else:
					#fire perceptron
					self.fired[i] = True	
					
				if self.fired[i] == self.train[i][2]:
					#if fired state is correct, log it
					self.correct[i] = True
					
				self.error[i] = self.fired[i]-self.train[i][2]
				
				for j in range(2):
					
					self.weights[j] += -self.learningRate*(self.error[0]**2+self.error[1]**2+self.error[2]**2)*self.train[i][j]
					self.bias += -self.learningRate*(self.error[0]+self.error[1]+self.error[2])*self.train[i][j]
				
				for j in range(2):
					print("W"+str(j)+": "+str(self.weights[j]))
				for j in range(3):
					print(self.fired[j])
				
	def Test(self):
		print("#----------------------------------#")
		#random point to test perceptron!
		if self.test[0]*self.weights[0]+self.test[1]*self.weights[1]++self.bias < self.threshold:
						#under thresh, dont fire
			self.testFired = False
		else:
					#fire perceptron
			self.testFired = True	
		print(self.testFired)
		for i in range(2):
			print(self.test[i])
		
		
			
#initialize object and call methods			
ptron = Train()
ptron.trainIt()
ptron.Test()
