import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelBinarizer

df_train = pd.read_csv('./MNIST_data_csv/mnist_train.csv', header=None)
df_test = pd.read_csv('./MNIST_data_csv/mnist_test.csv', header=None)

train_list = list(df_train)
test_list = list(df_test)

train_images = df_train.iloc[:,1:].values
test_images = df_test.iloc[:,1:].values

lb1 = LabelBinarizer()
lb2 = LabelBinarizer()
train_labels = lb1.fit_transform(df_train.iloc[:,0].values)
test_labels = lb2.fit_transform(df_test.iloc[:,0].values)

mnist = {'train':{
        'images':train_images,
        'labels':train_labels},
        'test':{
        'images':test_images,
        'labels':test_labels}
         }


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
        feed_train = {x: mnist['train']['images'], y: mnist['train']['labels'], dropout_rate: 0.7}
        feed_test = {x: mnist['test']['images'], y: mnist['test']['labels'], dropout_rate: 1}

        # training cycle
        for step in range(501):
                sess.run(optimizer, feed_dict=feed_train)

                if step % 20 == 0:
                        print(step, sess.run(cost, feed_dict=feed_train))
                if step % 100 == 0:
                        correc_prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correc_prediction, tf.float32))
                        print(sess.run(accuracy, feed_dict=feed_test))