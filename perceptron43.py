#não está funcionando
#https://gist.githubusercontent.com/audy/a7c7c3b31172fc8c2b685cc54bbb5ae8/raw/7cd36f415d9ee0cb29783627481aaf73c87f7553/perceptron.py

#!/usr/bin/env python3

class Perceptron:

    def __init__(self, weights, bias):

        self.weights = weights
        self.bias = bias

    def __getitem__(self, inputs):

        return int((sum(i*w for i, w in zip(inputs, self.weights)) + self.bias) > 0)


# build a NAND gate
nand_get = Perceptron([-2, -2], 3)

assert nand_gate[1, 1] == 0
assert nand_gate[0, 0] == 1
assert nand_gate[0, 1] == 1
assert nand_gate[1, 0] == 1
