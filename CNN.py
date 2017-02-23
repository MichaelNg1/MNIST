# Kaggle MNIST Exercise
# This CNN follows closely with the tensorflow tutorial

import numpy as np
import tensorflow as tf
import csv

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x,W):
  return tf.nn.conv2d(x,W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Parameters
BATCH_SIZE = 100

# Import all the training and testing data
test_data = np.genfromtxt('test.csv', skip_header=1, delimiter=',')
train_raw = np.genfromtxt('train.csv', skip_header=1, delimiter=',')
train_label = train_raw[:,0]
train_label_soft = np.zeros((len(train_label), 10))
for i in range( len(train_label) ):
	train_label_soft[i, int(train_label[i])] = 1
train_data = train_raw[:,1:train_raw.shape[1]]

print(train_label)

# Create the model
x = tf.placeholder(tf.float32, [None, train_data.shape[1]])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# Set up first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Set up the second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Densely Connected Layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_ = tf.placeholder(tf.float32, [None, 10])

# Lossfunction
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for i in range(20000):
	indices = np.random.permutation(range(train_data.shape[0]))[0:(BATCH_SIZE-1)]

	train_step.run(feed_dict={x: train_data[indices,:], y_: train_label_soft[indices], keep_prob: 0.5})
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: train_data[indices,:],
			y_: train_label_soft[indices], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy))

# Batch the test data and write to csv
with open('output1.csv','w', newline='') as f:
	writer = csv.writer(f)
	writer.writerow( ('ImageId', 'Label') )

	for start in range(0, test_data.shape[0], BATCH_SIZE):
		# Ensure we stay within boundaries
		end = np.minimum(start + BATCH_SIZE, test_data.shape[0])
		prediction = tf.argmax(y_conv,1)
		labels = prediction.eval(feed_dict={x: test_data[start:end,:], keep_prob: 1.0})
		for i in range(BATCH_SIZE):
			writer.writerow( (start + i + 1, labels[i]))

