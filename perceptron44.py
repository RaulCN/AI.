#https://gist.githubusercontent.com/JoshBroomberg/7b13267b7f80de2182a06daa2b14dba6/raw/dc134d357afb2d20c8ca848ec09f70aae7768cfd/and_perceptron.py
#Fiz algumas mudanças aparentemente eu concertei, falta traduzir para português e entender melhor o código
import random
import numpy as np 

a_weight = random.random()
b_weight = random.random()
bias_weight = random.random()
learning_constant = 1

valid_data = [(0,0,-1), (0,1,-1), (1,0,-1), (1,1,1)]

# Take inputs A and B
# Calculate an activation sum and then use the activation function
# to determine if perceptron should activate, simply
# return the value it provides.
def predict(a, b):
  activation_sum = a*a_weight + b*b_weight + bias_weight
  return activation_function(activation_sum)

# Take an activation sum and return 1 if that sum is greater than/equal 1
# -1 if it isn't. x >= 1 is the activation function.
def activation_function(activation_sum):
  if activation_sum >= 1:
    return 1
  else:
    return -1

# Take inputs and the error they produced and adjust the weights
# by input times error times learning_constant.
def adjust(a, b, error):
  global a_weight
  global b_weight
  global bias_weight
  
  a_weight += a * error * learning_constant
  b_weight += b * error * learning_constant
  bias_weight += error * learning_constant

# Return a random set of training data. A set is two inputs and an output.
def training_set():
  return random.sample(valid_data, 1)[0]

# This training function is pretty inefficient. 
# Basically, it gets a random training set, uses the perceptron to predict the value, and then
# adjust weights based on error between predicted val and the correct value.
# It will repeat this 1000 times or until the perceptron gets the output for all 4 possible
# inputs correct.
def train():
  attempts = 0

  while evaluate() < 4 and attempts < 1000:
    attempts += 1
  
    value_set = training_set()
    predicted = predict(value_set[0], value_set[1])
    if predicted != value_set[2]:
      error = value_set[2] - predicted
      adjust(value_set[0], value_set[1], error)
  
  print ("Number of training runs:"), attempts

# This function tests how many of the 4 possible input scenarios the current
# percepton would get right. It returns a number between 0 and 4.
def evaluate():
  correct = 0
  for values in valid_data:
    predicted = predict(values[0], values[1])
    if predicted == values[2]:
      correct += 1
  return correct

# This resets the perceptron.
def reset():
  global a_weight
  global b_weight
  global bias_weight
  global weights

  a_weight = random.random()
  b_weight = random.random()
  bias_weight = random.random()

def print_perceptron():
  print ("weights:", "a:", a_weight, "b:", b_weight, "bias:", bias_weight)
  print ("Accuracy:", evaluate(), "/4")
  print

# This function with reset and train the perceptron until a valid one is found. 
# It is necessary because sometimes the train function will exceed 1000 attempts
# which means it will stop before the perceptron is valid.
def find_a_valid_perceptron():
  attempts = 0
  while evaluate() != 4 and attempts < 10000:
    attempts += 1
    reset()
    train()

    print ("Attempt number:"), attempts
    print_perceptron()

  print ("Done. Valid perceptron:")
  print_perceptron()

find_a_valid_perceptron()
