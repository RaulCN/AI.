#não está funcionando
#https://gist.githubusercontent.com/ogasawaraShinnosuke/529a0b3bc6425d4b45093033da27700e/raw/a453b4041ab5c206843401dd100e23c6571a6272/perceptron.py

import numpy as np

single_perceptron = lambda x, w, theta: 0 if np.sum(w * x) + theta <= 0 else 1
and_perceptron = lambda x: single_perceptron(x, np.array([0.5, 0.5]), -0.7)
nand_perceptron = lambda x: single_perceptron(x, np.array([-0.5, -0.5]), 0.7)
or_perceptron = lambda x: single_perceptron(x, np.array([0.5, 0.5]), -0.2)
xor_perceptron = lambda x: and_perceptron(np.array([nand_perceptron(x), or_perceptron(x)]))

INPUT_DATA = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])

print ("> AND")
print ([and_perceptron(i) for i in INPUT_DATA])
print ("> NAND")
print ([nand_perceptron(i) for i in INPUT_DATA])
print ("> OR")
print ([or_perceptron(i) for i in INPUT_DATA])
print ("> XOR")
print ([xor_perceptron(i) for i in INPUT_DATA])
