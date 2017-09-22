#https://gist.github.com/kyleholzinger/c364ff76d9b08f42094d
#não funciona

"""Learning and prediction functions for perceptrons."""

# NAME: Kyle Holzinger
# BUID: U92663004
# DATE: 9.4.14

import common


class NotConverged(Exception):
    """An exception raised when the perceptron training isn't converging."""


class Perceptron(object):
    """ Learns and predicts based on test data. """
    def __init__(self, weights=None):
        self.sum = 0
        self.weights = weights

    def learn(self, examples, max_iterations=100):
        """
        Learn a perceptron from [([feature], class)].

        Set the weights member variable to a list of numbers corresponding
        to the weights learned by the perceptron algorithm from the training
        examples.

        The number of weights should be one more than the number of features
        in each example.

        Args:
          examples: a list of pairs of a list of features and a class variable.
            Features should be numbers, class should be 0 or 1.
          max_iterations: number of iterations to train.  Gives up afterwards

        Raises:
          NotConverged, if training did not converge within the provided number
            of iterations.

        Returns:
          This object
        """
        # COMPLETE THIS IMPLEMENTATION

        # Set up self.weights. Initialize them to 0.
        self.weights = []
        num_units = len(examples[0][0])
        for _ in xrange(num_units):
            self.weights.append(0)
        cur_itteration = 0
        done = True

        # update loop
        while True:
            done = True
            for example in examples:
                if self.predict(example[0]) != example[1]:
                    done = False
                if self.predict(example[0]) is 0 and example[1] is 1:
                    common.scale_and_add(self.weights, 1, example[0])
                else:
                    common.scale_and_add(self.weights, -1, example[0])
            if done is True:
                break
            cur_itteration += 1
            if cur_itteration >= max_iterations:
                raise NotConverged
        return self

    def predict(self, features):
        """Return the prediction given perceptron weights on an example.

        Args:
          features: A vector of features, [f1, f2, ... fn], all numbers

        Returns:
          1 if w1 * f1 + w2 * f2 + ... * wn * fn + t > 0
          0 otherwise
        """
        # COMPLETE THIS IMPLEMENTATION
        # sum = 0
        # for f, w in zip(features, self.weights):
        #   sum = sum + f * w
        # features.append (1)
        _tt = 0
        if len(features) is not len(self.weights):
            _tt = self.weights[len(self.weights) - 1]
            self.weights.remove(_tt)
        self.sum = common.dot(features, self.weights)
        self.sum += _tt
        return common.step(self.sum)
