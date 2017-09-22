#funciona falta entender melhor
#https://gist.github.com/ebraminio/77108bcc297adfda1d76c9cb71b34fa0

# `time python3 perceptron.py`
# real    0m6.740s
# user    0m6.580s
# sys     0m0.044s
# `time pypy perceptron.py`
# real    0m0.593s
# user    0m0.536s
# sys     0m0.049s
import random

speed = 0.01
num_weights = 3
weights = []

def feed_forward(inputs):
    sum = 0
    # multiply inputs by weights and sum them
    for x in range(0, len(weights)):
        sum += weights[x] * inputs[x]
    # return the 'activated' sum
    return activate(sum)

def activate(num):
    # turn a sum over 0 into 1, and below 0 into -1
    if num > 0:
        return 1
    return -1

def train(inputs, desired_output):
    guess = feed_forward(inputs)
    error = desired_output - guess

    for x in range(0, len(weights)):
        weights[x] += error * inputs[x] * speed

for x in range(0, num_weights):
    weights.append(random.random()*2-1)

for x in range(0, 1000000):
    x_coord = random.random()*500-250
    y_coord = random.random()*500-250
    line_y = .5 * x_coord + 10 # line: f(x) = 0.5x + 10

    if y_coord > line_y: # above the line
        train([x_coord, y_coord, 1], 1)
    else: # below the line
        train([x_coord, y_coord, 1], -1)

print("(-7, 9): " + str(feed_forward([-7,9,1])))
print("(3, 1): " + str(feed_forward([3,1,1])))
print(weights)
