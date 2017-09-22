#Não está funcionando
#https://gist.githubusercontent.com/carlomazzaferro/f2ad440ad45684485d386067483ff12c/raw/546e5e3739943d5bc6aed2c215633361306bb7c8/perceptron.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Perceptron(object):

    def __init__(self, train, labels, learning_rate, iterations):

        self.train = train
        self.labels = labels
        self.weights = numpy.random.rand(len(self.train[0])) - 0.5  # some rand values
        self.bias =  numpy.random.rand() - 0.5
        self.max_iter = iterations
        self.l_rate = learning_rate
        self.total_error = 0
        self.round_errors = []  # for plotting

    def train_weights(self):

        _learned = False
        _round = 1

        while not _learned:

            self.total_error = 0

            for idx, row in enumerate(self.train):
                prediction = self.predict(row)
                error = self.labels[idx] - prediction
                self.update_weights(row, error)
                self.update_total_error(error)
                self.update_bias(error)

            self.round_errors.append((_round, abs(self.total_error)))

            print('Round=%i, lrate=%.3f, error=%.3f' % (_round, self.l_rate, abs(self.total_error)))

            _round += 1
            if self.total_error == 0.0 or _round >= self.max_iter:  # stop criteria
                print('# Rounds', _round)
                _learned = True  # stop learning

    def update_bias(self, error):
        self.bias += self.l_rate * error

    def update_total_error(self, error):
        self.total_error += error

    def update_weights(self, row, err):
        for i in range(len(self.weights)):
            self.weights[i] += self.l_rate * err * row[i]

    # dot product of weigths and x
    def predict(self, row):
        output = 0
        for i in range(len(self.weights)):
            output += self.weights[i] * row[i] + self.bias
        return self.determine_pred(output)

    @staticmethod
    def determine_pred(output):
        return 1.0 if output >= 0.0 else -1.0

    @staticmethod
    def plot(err):
        iteration = np.array([e[0] for e in err])
        error = np.array([e[1] for e in err])
        plt.plot(iteration, error)
        plt.xlabel('Round Number')
        plt.ylabel('Absolute Error')
        plt.title('Total Error By Round')
        plt.show()


if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = digest('Q1_data.txt')
    pcp = Perceptron(X_train, Y_train, 0.001, 100)
    pcp.train_weights()

    print('Weights: %f, %f, %f' % (pcp.weights[0], pcp.weights[1], pcp.weights[2]))
    print('Bias: %f' % pcp.bias)
    pcp.plot(pcp.round_errors)   #shown in report

    #Outputs:

    """
    Round=1, lrate=0.001, error=6.000
    Round=2, lrate=0.001, error=2.000
    Round=3, lrate=0.001, error=2.000
    Round=4, lrate=0.001, error=2.000
    Round=5, lrate=0.001, error=0.000
    # Rounds 6
    Weights: 0.559600, -0.471400, 0.141200
    Bias: -0.688000
    """
    #Predict and calculate test error

    test_error = 0
    for i, row in enumerate(X_test):
        test_error += abs(pcp.predict(row) - Y_test[i][0])
    print('Cumulative Test Error: %f' % test_error)

    #Output
    """
    Cumulative Test Error: 0.000000
    """


