import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelBinarizer

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs+n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0/ (n_inputs+n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


# parameter
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# tf graph input
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

# 기본 초기값
# w1 = tf.Variable(tf.random_normal([784,256]))
# w2 = tf.Variable(tf.random_normal([256,256]))
# w3 = tf.Variable(tf.random_normal([256,10]))

# Xavier initialization
w1 = tf.get_variable('w1', shape=[784, 256], initializer=xavier_init(784, 256))
w2 = tf.get_variable('w2', shape=[256, 256], initializer=xavier_init(256, 256))
w3 = tf.get_variable('w3', shape=[256, 128], initializer=xavier_init(256, 128))
w4 = tf.get_variable('w4', shape=[128, 64], initializer=xavier_init(128, 64))
w5 = tf.get_variable('w5', shape=[64, 10], initializer=xavier_init(64, 10))
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([128]))
b4 = tf.Variable(tf.random_normal([64]))
b5 = tf.Variable(tf.random_normal([10]))

# 기본 모델
# l1 = tf.nn.relu(tf.matmul(x,w1)+b1)
# l2 = tf.nn.relu(tf.matmul(l1,w2)+b2) # hidden layer with relu activation
# hypothesis = tf.matmul(l2,w3)+b3 # no need to use softmax here

# more deep & Dropout
dropout_rate = tf.placeholder('float')
_l1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
l1 = tf.nn.dropout(_l1, dropout_rate)
_l2 = tf.nn.relu(tf.add(tf.matmul(l1, w2), b2))
l2 = tf.nn.dropout(_l2, dropout_rate)
_l3 = tf.nn.relu(tf.add(tf.matmul(l2, w3), b3))
l3 = tf.nn.dropout(_l3, dropout_rate)
_l4 = tf.nn.relu(tf.add(tf.matmul(l3, w4), b4))
l4 = tf.nn.dropout(_l4, dropout_rate)

hypothesis = tf.add(tf.matmul(l4, w5), b5)

# define cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y))  # softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initializing the variables
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # training cycle <- mnist database 있을 때,
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7})
            # compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7}) / total_batch

        if epoch % display_step == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
            correc_prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correc_prediction, tf.float32))
            print('Accuracy: ',
                  sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1.0}))

    print('Optimization Finished!')