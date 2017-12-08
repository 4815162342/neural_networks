#!/usr/bin/env python

import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 101)  # generate x values from -1 to 1 with 99 values in between
trY = 2 * trX + np.random.randn(101) * 0.33  # create a y value which is approximately linear but with some random noise

X = tf.placeholder("float")  # create symbolic variables -- placeholder
Y = tf.placeholder("float")  # create symbolic variables -- placeholder

def model(X, w):
	return tf.multiply(X, w)  # like a system of equations, weights multiplying data


w = tf.Variable(0.0, name="weights")  # create a shared variable for the weight matrix
y_model = model(X, w)

cost = tf.square(Y - y_model)  # use square error for cost function

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)  # construct an optimizer to minimize cost and fit line to my data

# Launch the graph in a session
with tf.Session() as sess:
	# you need to initialize variables (in this case just variable W)
	tf.global_variables_initializer().run()

	for i in range(100):
		for (x, y) in zip(trX, trY):
			sess.run(train_op, feed_dict={X: x, Y: y})

	print(sess.run(w))  # It should be something around 2