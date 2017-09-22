#estáfuncionando
#Fonte: https://gist.github.com/nicolehe/d02d1801841252fd16eb47ba999c2cf0

import numpy as np
#test script for implementing perceptron
N = 4
complete = False

def calc(test, weight):
    array_after = weight * test
    res = np.zeros(N)
    index = 0
    num = 0
    while num < N:
        data = array_after[num,0] + array_after[num, 1]
        res[num] = data
        num+=1
    return array_after, res

def isTrue(res):
    if res[0] < 0:
        return 0
    elif res[1] < 0:
        return 1
    elif res[2] > 0:
        return 2
    elif res[3] > 0:
        return 3
    else:
        return -1

def train(test, weight, index):
    if index < 2:
        print ("calculate new weight for class 1")
        weight_new = weight + 0.5 * test
        print (weight_new)
    else:
        print ("calculate new weight for class 2")
        weight_new = weight - 0.5 * test
        print (weight_new)
    return weight_new

# weght and test data
weight = np.array([0.2, 0.3])
test = np.array([[1.0, 1.0], [1.0, 0.5], [1.0, -0.2], [1.0, -1.3]])

err = 0

while err >= 0:
    array_after, res = calc(test, weight)

    err = isTrue(res)
    if err != -1:
        weight = train(test[err], weight, err)

print ("result is")
print (array_after)

print (res)
