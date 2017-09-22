#original https://gist.githubusercontent.com/iamPres/fdd98cd14b0fc3c35f265c5b57de6fd3/raw/3f0474b4eb9c35ccaa7d9441a6d2cedcee9fd0d9/perceptron.py#
import random

class Train:	
	def __init__(self):
		self.train = [[10,5,3,False],[3,2,16,True],[4,15,11,False]]
		#Initialize training set
		self.fired = False #activate when over thresh 
		self.threshold = 1.0 
		self.x1 = 1 #weights
		self.x2 = -1 
		self.x3 = 0.5
		self.learningRate = 0.1 #not used
	#true if fired state == answers in training set
		self.correct = [False,False,False] 
		self.desire ='' #input value
	
	def trainIt(self):	
		#while one or more of the answers are wrong
		while self.correct[1] == False or self.correct[0] == False or self.correct[2] == False:
			for i in range(3):
				#Perceptron formula
				if self.train[i][0]*self.x1+self.train[i][1]*self.x2+self.train[i][2]*self.x3 < self.threshold:
					#under thresh, dont fire
					self.fired = False
				else:
					#fire perceptron
					self.fired = True	
					
				if self.fired == self.train[i][3]:
					#if fired state is correct, log it
					self.correct[i] = True
				else:
					#my subsitution for the learning rate
					self.x1 += random.randrange(-1,2)
					self.x2 += random.randrange(-1,2)
					self.x3 += random.randrange(-1,2)
			#prints weights
			print("W1: "+str(self.x1))
			print("W2: "+str(self.x2))
			print("W3: "+str(self.x3))
						
		self.Test()
				
	def Test(self):
		#random point to test perceptron!
		self.test = [3,20,8]
		#reset some values
		self.fired = False
		self.correct = [False,False,False]
		
		#fire if over threshold
		if self.test[0]*self.x1+self.test[1]*self.x2 < self.threshold:
			self.fired = False
		else:
			self.fired = True
			
		print("Test Result: "+str(self.fired))
		self.desire = input("Was this your desired result? ")
		#print the test result and ask the user if the perceptron was right. This would be useful in filtering spam
		if self.desire == 'N':
			#if it was not right, continue training
			print("Okay! Training...")
			self.trainIt()
			
#initialize object and call methods			
ptron = Train()
ptron.trainIt()
ptron.Test()
