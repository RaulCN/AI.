#Uma rede neural simples linhas de Python 
#https://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
y = np.array([[0,0,1,1]]).T
np.random.seed(1)
syn0 = 2*np.random.random((3,1)) - 1
for iter in range(10000): #xrange n existe mais no python 3
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1,True)
    syn0 += np.dot(l0.T,l1_delta)
print ("Output After Training:") #o código original n tem parênteses
print (l1) #o código original n tem parênteses

#apenas adaptei para a nova versão de python 3
