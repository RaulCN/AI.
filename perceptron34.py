#precisa do tensor flow ainda não testei
#https://gist.githubusercontent.com/ClaudeCoulombe/da1e1de3c0a25c41ab3a95000ca149b1/raw/ef9b43e853cd12e5fab8017e52cc838db95cbbf2/gistfile1.txt

# 6.1 Example: Learning XOR - GBC Book - Chapter 6 - pp. 166 to 171
# Some parts are inspired by the blog post
# Solving XOR with a Neural Network in TensorFlow
# by Stephen OMAN
# https://github.com/StephenOman/TensorFlowExamples/blob/master/xor%20nn/xor_nn.py

# Activation RELU + sigmoid for binary classification output + MSE loss function
import tensorflow as tf
import time
import numpy as np

X = tf.placeholder(tf.float32, shape=[4,2], name = 'X')
Y = tf.placeholder(tf.float32, shape=[4,1], name = 'Y')

W = tf.Variable(tf.truncated_normal([2,2]), name = "W")
w = tf.Variable(tf.truncated_normal([2,1]), name = "w")

c = tf.Variable(tf.zeros([4,2]), name = "c")
b = tf.Variable(tf.zeros([4,1]), name = "b")

with tf.name_scope("hidden_layer") as scope:
    h = tf.nn.relu(tf.add(tf.matmul(X, W),c))

with tf.name_scope("output") as scope:
    y_estimated = tf.sigmoid(tf.add(tf.matmul(h,w),b))

with tf.name_scope("loss") as scope:
    loss = tf.reduce_mean(tf.squared_difference(y_estimated, Y)) 

# For better result with binary classifier, use cross entropy with a sigmoid
#    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_estimated, labels=Y)

# A naïve direct implementation of the loss function
#     n_instances = X.get_shape().as_list()[0]
#     loss = tf.reduce_sum(tf.pow(y_estimated - Y, 2))/ n_instances

# In case of problem with gradient (exploding or vanishing gradient)perform gradient clipping
#     n_instances = X.get_shape().as_list()[0]
#     loss = tf.reduce_sum(tf.pow(tf.clip_by_value(y_estimated,1e-10,1.0) - Y,2))/(n_instances)

with tf.name_scope("train") as scope:
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

INPUT_XOR = [[0,0],[0,1],[1,0],[1,1]]
OUTPUT_XOR = [[0],[1],[1],[0]]

init = tf.global_variables_initializer()
sess = tf.Session()

writer = tf.summary.FileWriter("./logs/xor_logs", sess.graph)

sess.run(init)

t_start = time.clock()
for epoch in range(100001):
    sess.run(train_step, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR})
    if epoch % 10000 == 0:
        print("_"*80)
        print('Epoch: ', epoch)
        print('   y_estimated: ')
        for element in sess.run(y_estimated, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR}):
            print('    ',element)
        print('   W: ')
        for element in sess.run(W):
            print('    ',element)
        print('   c: ')
        for element in sess.run(c):
            print('    ',element)
        print('   w: ')
        for element in sess.run(w):
            print('    ',element)
        print('   b ')
        for element in sess.run(b):
            print('    ',element)
        print('   loss: ', sess.run(loss, feed_dict={X: INPUT_XOR, Y: OUTPUT_XOR}))
t_end = time.clock()
print("_"*80)
print('Elapsed time ', t_end - t_start)
