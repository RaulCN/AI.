#acho q está funcionando, mas o que ele faz?
#https://gist.github.com/drcoelho/3db229d3e8d00dac25b8
import numpy as np

ALPHA = .1
LOOPS = 1000


def perceptron_learning_algorithm(X, Y, threshold=5):
    counter = 0
    w = np.zeros(2)

    converged = False
    while not converged and counter <= LOOPS:

        sign = lambda x: 1 if x >= 0 else -1

        counter += 1
        converged = True

        for index, x in enumerate(X):

            expected = Y[index]
            hypothesis = sign(np.inner(w, x) - threshold)

            error = expected - hypothesis

            w += ALPHA * error * x
            if error != 0:
                converged = False
                break

    return w, converged, counter


if __name__ == '__main__':

    BAD_CUSTOMERS = np.array([
        (2.0, 1.0),
        (3.0, 0.5),
        (3.5, 0.5),
        (1.0, 1.5),
        (4.0, 0.5)])

    GOOD_CUSTOMERS = np.array([
        (1.5, 8.0),
        (0.5, 9.0),
        (3.8, 4.9),
        (3.0, 5.1),
        (4.5, 1.5)])

    Y1 = np.ones(len(GOOD_CUSTOMERS))
    Y2 = np.ones(len(BAD_CUSTOMERS)) * -1.

    X = np.concatenate((GOOD_CUSTOMERS, BAD_CUSTOMERS))
    Y = np.concatenate((Y1, Y2))

    print (perceptron_learning_algorithm(X, Y))
