#! /usr/bin/python
#n funciona
data = [[1, 1], [1, 0], [0, 1], [0, 0]]
target = [1, -1, -1, -1]
weight = [0, 0]
bias = 0
learning_rate = 1
activator = 0
theta = 0.2
epoch = 1

while activator == 0:
	fnet_list = []
	bias_list = []

	print "[*] Reaching Epoch: " + str(epoch)

	for i in range(len(data)):
		y_in = bias + reduce(lambda a, b: a + b, map(lambda (j, x): x * weight[j], enumerate(data[i])))

		if y_in < -theta:
			f_net = -1
		elif y_in >= -theta and y_in <= theta:
			f_net = 0
		elif y_in > theta:
			f_net = 1

		old_weight = weight
		old_bias = bias

		fnet_list.append(f_net)

		if f_net != target[i]:
			weight = map(lambda (j, x): x + (learning_rate * (data[i][j] * target[i])), enumerate(weight))
			bias = old_bias + (learning_rate * target[i])
		else:
			weight = old_weight
			bias = old_bias

		bias_list.append(bias - old_bias)
 
	if fnet_list == target:
		activator = 1

	epoch = epoch + 1
