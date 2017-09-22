#fonte https://gist.githubusercontent.com/nicolehe/d02d1801841252fd16eb47ba999c2cf0/raw/6a3a14f4b4891039f31de221824ff6a518ff090d/perceptron.py
#corri problemas de aspas mas ainda n funciona
import numpy as np
import random
import sys

and_gate = [
# [(inputs), expected output]
    [(1, 1), 1],
    [(1, -1), -1],
    [(-1, 1), -1],
    [(-1, -1), -1]
]

or_gate = [
    [(1, 1), 1],
    [(1, -1), 1],
    [(-1, 1), 1],
    [(-1, -1), -1]
]

def activation_function(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1

def run_perceptron(gate):
    bias = (1,) # the bias is always one
    learning_constant = 0.1
    n = 50 # how many times the machine learns

    weights = []

    # initialize with 3 random weights between -1 and 1, one for each input and one for the bias
    for i in range(3):
        weights.append(random.uniform(-1, 1))

    for i in range(n):
        inputs, expected_output = random.choice(gate)
        inputs = inputs + bias # add the bias here
        weighted_sum = np.dot(inputs, weights)
        guess = activation_function(weighted_sum) # find the sign of the weighted sum
        error = expected_output - guess
        weights += learning_constant * error * np.asarray(inputs) # change the weights to include the error times input, won't change if there's no error


    inputs, expected_output = random.choice(gate)
    print ("inputs: " + str(inputs))
    inputs = inputs + bias
    weighted_sum = np.dot(inputs, weights)
    print ("weighted sum: " + str(weighted_sum))
    print ("correct answer: " + str(expected_output))
    print ("perceptron guess: " + str(activation_function(weighted_sum)) + '\n')

tests = int(sys.argv[1])

for i in range(tests):
    print ("// AND //")
    run_perceptron(and_gate)

    print ("// OR //")
    run_perceptron(or_gate)

# to run the program, type in 'python perceptron.py' followed by the number of tests you want to see for each
# for example:
# python perceptron.py 2
#
#
# // AND //
# inputs: (1, 1)
# weighted sum: 0.49538226743
# correct answer: 1
# perceptron guess: 1
#
# // OR //
# inputs: (1, 1)
# weighted sum: 1.55188314007
# correct answer: 1
# perceptron guess: 1
#
# // AND //
# inputs: (-1, -1)
# weighted sum: -1.30320106541
# correct answer: -1
# perceptron guess: -1
#
# // OR //
# inputs: (1, 1)
# weighted sum: 1.31615731054
# correct answer: 1
# perceptron guess: 1
