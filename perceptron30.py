#Fiz alguns concertos mas ainda não está funcionando
#https://gist.githubusercontent.com/Millsky/4285d6a98465b70023385896a8502e4a/raw/3d7b9bdb6690bee27b6b135b2e5c72223a51fa8e/perceptron.py

import numpy as np 

inputs  =  np.array([[1],[100],[3],[30],[40]])
testSet =  np.array([[10],[13],[19],[33],[1]])
outputs =  np.array([[5],[203],[9],[63],[83]])
bias    =  1
biasWeight = 1
#initWeights 
weights = np.random.random()

print (inputs.shape)


def sigmoid(x):
	return 1/(1+np.exp(-x))

for j in range(1, 70000):
	for k in range(len(inputs)):
		x = (inputs[k] * weights) + (1 * biasWeight)
		x = sigmoid(x)
		biasWeight = biasWeight + sigmoid(outputs[k]) - x
		weights = weights + ((sigmoid(outputs[k]) - x) * inputs[k])
		if j%10000 == 0:
			print (weights)
	
print ((testSet * weights) + (bias * biasWeight))
