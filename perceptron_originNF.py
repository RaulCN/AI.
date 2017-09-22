#!/usr/bin/env python    ##Não funciona
# -*- coding: utf-8 -*-
#original https://gist.github.com/Linusp/348b20c06d856bc6271b#file-perceptron_origin-py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

x = np.loadtxt('a.txt') #como carrega isso?

weight = np.array([0.0, 0.0])
bias = 0.0
fai = 1.0

def is_error_classify(arr):
    result = (np.dot(arr[0:2], weight) + bias) * arr[2]
    return result <= 0

def first_error():
    global x
    for i in range(0, len(x)):
        if is_error_classify(x[i]):
            return i
    return -1

def update(i):
    global weight, bias, x
    weight = weight + x[i][0:2] * x[i][2] * fai
    bias = bias + x[i][2] * fai
    print ("Error points(%d): (%f %f);Weight: (%f %f - %d); bias: %f" %(i,
    x[i][0], x[i][1], x[i][2], weight[0], weight[1], bias))

def plot_all_points():
    global x, ax1
    for point in x:
        if point[2] == 1:
            ax1.plot(point[0], point[1], ('ro')
        else:
            ax1.plot(point[0], point[1], ('bo')


def animate(i):
    global weight, bias, x
    err_index = first_error()
    if err_index >= 0:
        update(err_index)
    else:
        exit
    ax1.clear()
    plot_all_points()
    x_range = np.arange(-5, 15, 0.025)
    y_range = np.arange(-5, 15, 0.025)
    X, Y = np.meshgrid(x_range, y_range)
    f = weight[0] * X + weight[1] * Y + bias
    ax1.contour(X, Y, f, [0], colors=('green'))
    plt.xlim([-5, 15])
    plt.ylim([-5, 15])

if __name__ == '__main__':
    plot_all_points()
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()
