'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
import data
# Parameters
learning_rate = 0.001
training_iters = 10
batch_size = 40
display_step = 1

# Network Parameters
n_input = 2679 # MNIST data input (img shape: 28*28)
n_classes = 40 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
	# Conv2D wrapper, with bias and relu activation
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)


def maxpool2d(x, k=2):
	# MaxPool2D wrapper
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='VALID')


# Create model
def conv_net(x, weights, biases, dropout):
	# Reshape input picture
	x = tf.reshape(x, shape=[-1, 57, 47, 1])

	# Convolution Layer 1 
	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	# Max Pooling (down-sampling)
	conv1 = maxpool2d(conv1, k=2)

	# Convolution Layer 2
	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	# Max Pooling (down-sampling)
	conv2 = maxpool2d(conv2, k=2)
   
	# Fully connected layer 
	# Reshape conv2 output to fit fully connected layer input
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	# Apply Dropout
	fc1 = tf.nn.dropout(fc1, dropout)

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out

# Store layers weight & bias
weights = {
	# 5x5 conv, 1 input, 32 outputs
	'wc1': tf.Variable(tf.random_normal([5, 5, 1, 4])),
	# 5x5 conv, 32 inputs, 64 outputs
	'wc2': tf.Variable(tf.random_normal([5, 5, 4, 8])),
	# fully connected, 7*7*64 inputs, 1024 outputs
	'wd1': tf.Variable(tf.random_normal([14*11*8, 2679])),
	# 1024 inputs, 10 outputs (class prediction)
	'out': tf.Variable(tf.random_normal([2679, n_classes]))
}

biases = {
	'bc1': tf.Variable(tf.random_normal([4])),
	'bc2': tf.Variable(tf.random_normal([8])),
	'bd1': tf.Variable(tf.random_normal([2679])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



# Visual the data
cost_summary = tf.scalar_summary("cost", cost)
acc_summary = tf.scalar_summary("accuracy", accuracy)
merged_summary_op = tf.merge_summary([cost_summary, acc_summary])

# Initializing the variables
init = tf.global_variables_initializer()


# Launch the graph
with tf.Session() as sess:
	sess.run(init)
	summary_write = tf.train.SummaryWriter('tmp/', sess.graph)
	# Keep training until reach max iterations
	total_batch = 320/batch_size
	print ("Training start!")
	for iters in range(training_iters):
		avg_cost = 0
		for i in range(total_batch):
		# Run optimization op (backprop)
			batch_x, batch_y = data.return_next_batch(batch_size, i)
			# print (tf.argmax(batch_y, 1))
			loss, acc, _, summary_str = sess.run([cost, accuracy, optimizer, merged_summary_op], feed_dict={x: batch_x, y: batch_y,
									keep_prob: dropout})
			summary_write.add_summary(summary_str, iters+1)

			avg_cost += loss / total_batch
		if (iters+1) % display_step == 0:
			# Calculate batch loss and accuracy
			print("Iter ", '%04d' % (iters+1), ", Average Loss= " + \
				"{:.6f}".format(avg_cost), ", Training Accuracy= ", \
				"{:.5f}".format(acc))

			#batch_x, batch_y = data.return_next_batch(0, 0)
			#summary_str = sess.run(merged_summary_op, feed_dict = {x: batch_x, y: batch_y})

	print("Optimization Finished!")

	#test_x, test_y = data.return_next_batch(400, 0)
	test_x, test_y = data.return_test_data()
	print("Testing Accuracy:", \
	sess.run(accuracy, feed_dict={x: test_x, y: test_y, keep_prob: 1.}))
	summary_write.close()
