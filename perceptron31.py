"""Simple Perceptron implementation.

No 3rd-party modules (numpy) used.
"""
from math import copysign


def sgn(x):
    """Mathemetical sign function implementation.

    return 1 for x >= 0 or -1 otherwise.
    """
    return copysign(1, x)


def classify(weights, bias, sample):
    """Classifiy a sample according to given weights.

    We do this by calculating the dot-product wTx >= 0.
    Sometimes also denoted as <weights, sample> >= 0.

    weights: The weights vector.
    bias: The weights vector's bias.
    sample: The sample.

    weights and sample are two same-sized lists.
    """
    # Calculate sum(wi*si) for i=0 to N, and add the bias at the end.
    # Then return sgn() of result.
    return sgn(
        sum(weight * feature for weight, feature in zip(weights, sample))
        + bias)


def train(samples, labels, max_iter=100, rate=0.1):
    """Train perceptron on data, and returns a w in R^n vector.

    samples & labels are two same-sized lists of training samples and labels,
    sometimes denoted by (X, Y) in mathematical books.
    Each sample X[i] is labeled by Y[i].

    max_iter sets the maximum amount of iterations for the learning algorithm.

    rate sets the learning rate of the algorithm.
    That is, how strong to adjust the classifier on every iteration.
    """
    # Number of features (The dimension, that is, the N in R^N).
    dim = len(samples)
    weights = [0] * dim  # Weights vector.
    bias = 0

    for t in xrange(max_iter):  # Maximum of `max_iter` iterations.
        errors = 0
        for sample, label in zip(samples, labels):
            # Update weights, bias, and increase error count
            # if we labeled wrong.
            if classify(weights, bias, sample) != label:
                weights = [weight + (rate * label * feature)
                           for weight, feature in zip(weights, sample)]

                bias += rate * label

                errors += 1

        # Stop learning if perceptron labeled all samples without errors.
        if errors == 0:
            break

    # Return classifier function, which receives a sample x,
    # and returns its sign.
    return lambda x: classify(weights, bias, x)


def test(samples, labels, classifier):
    """Test classifier on samples, and returns error/total percentage."""
    errors = sum(1 for sample, label in zip(samples, labels)
                 if classifier(sample) != label)

    return 100.0 * errors / len(samples)
