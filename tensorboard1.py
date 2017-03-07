import tensorflow as tf
import numpy as np

xy = np.loadtxt('data_07.txt', unpack=True)
x_data = xy[0:-1]
y_data = xy[-1]

x_data_t = x_data.transpose()
y_data_t = y_data.transpose()
y_data_t =  [[0],[1],[1],[0]]


x_ = tf.placeholder(tf.float32, shape=[4, 2], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[4, 1], name="y-input")

w1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='w1')
w2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='w2')
b1 = tf.Variable(tf.zeros([2]), name='bias1')
b2 = tf.Variable(tf.zeros([1]), name='bias2')


# hypothesis
with tf.name_scope('layer1') as scope:
    l2 = tf.sigmoid(tf.matmul(x_, w1) + b1)
with tf.name_scope('last') as scope:
    hypothesis = tf.sigmoid(tf.matmul(l2, w2) + b2)

# cost
with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(y_ * tf.log(hypothesis) + (1 - y_) * tf.log(1 - hypothesis))
    cost_sum = tf.scalar_summary('cost',cost)

# train
with tf.name_scope('train') as scope:
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)


sess = tf.Session()

merged = tf.merge_all_summaries()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log/nntest", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(10001):
        sess.run(train_step, feed_dict={x_: x_data_t, y_: y_data_t})

        # fit the line
        if (i % 500 == 0):
            print(i, sess.run(cost, feed_dict={x_: x_data_t, y_: y_data_t}))
            summary = sess.run(merged, feed_dict={x_: x_data_t, y_: y_data_t})
            writer.add_summary(summary, i)



