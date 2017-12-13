# Kaggle MNIST Exercise

import numpy as np
import tensorflow as tf

# Parameters
BATCH_SIZE = 100

# Import all the training and testing data
test_data = np.genfromtxt('train.csv', skip_header=1, delimiter=',')
train_raw = np.genfromtxt('train.csv', skip_header=1, delimiter=',')
train_label = train_raw[:,0]
train_label_soft = np.zeros((len(train_label), 10))
for i in range( len(train_label) ):
	train_label_soft[i, int(train_label[i])] = 1
train_data = train_raw[:,1:train_raw.shape[1]]

print(train_label)

# Create the model
x = tf.placeholder(tf.float32, [None, train_data.shape[1]])
W = tf.Variable(tf.zeros([train_data.shape[1], 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

# The raw formulation of cross-entropy,
#
#   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
#                                 reduction_indices=[1]))
#
# can be numerically unstable.
#
# So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
# outputs of 'y', and then average across the batch.
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
#tf.global_variables_initializer().run()
tf.initialize_all_variables().run()

# Train
for _ in range(1000):
	for start in range(0, train_data.shape[0], BATCH_SIZE):
		end = start + BATCH_SIZE
		sess.run(train_step, feed_dict={x: train_data[start:end,:], y_: train_label_soft[start:end]})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: train_data,
                                  y_: train_label_soft}))
