"""
    convolutional neural network

        flow of the convolutional network
            for example, 256*256 RGB photo, the width and height are compressed into smaller 
            size of arrays and the depth is increased during 

            stride: the pixels for one step

            2 key methods: 
                1. valid padding: the resultant size after striding > original size
                2. same padding: the resultant size after striding = original size

            pooling is for solving large stride which will make the loss of the msg of orginal 
            photo. 
                Pooling: 
                    max pooling
                    avg pooling

            images -> convolution -> max pooling -> convolution -> max pooling -> fully connected
            -> fully connected -> classifier
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)
train_size = 500
acc_no = 1000
learning_rate = 0.0001
def compute_accuracy(v_xs, v_ys):
    global prediction
    global acc_no
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    # print('result: {}'.format(np.argmax(v_ys, 1)))
    # print('predic: {}'.format(np.argmax(y_pre, 1)))
    return result


xs = tf.placeholder(tf.float32, [None, 784], name='x_in')
ys = tf.placeholder(tf.float32, [None, 10], name='y_in')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
x_image = tf.reshape(xs, [-1, 28, 28, 1])
#print(x_image.shape)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def conv_layer(input, channel_in, channel_out, strides = 2, name='conv'):
    with tf.name_scope(name):
        # shape=[patch x's size, patch y'size, input height , output height]
        with tf.name_scope('Weight'):
            W_conv = tf.Variable(tf.truncated_normal(shape=[5,5,channel_in,channel_out], stddev = 0.1), name = 'W') #patch 5*5
            tf.summary.histogram(name + '/Weights', W_conv)
        with tf.name_scope('Bias'):
            b_conv = tf.Variable(tf.constant(0.1, shape =[channel_out]), name = 'B')
            tf.summary.histogram(name + '/Bias', b_conv)


        with tf.name_scope('conv_layer'):
            h_conv =tf.nn.relu(tf.nn.conv2d(input, W_conv, strides=[1,1,1,1], padding='SAME') + b_conv) #28*28*32

        with tf.name_scope('Pooling_layer'):
            h_pool = tf.nn.max_pool(h_conv, ksize = [1,2,2,1],strides=[1,strides,strides,1], padding='SAME')          #14*14*32

        return h_pool

def fc_layer(input, channel_in, channel_out, name='fc_layer'):
    with tf.name_scope(name):
        with tf.name_scope('Weight'):
            W_f = tf.Variable(tf.truncated_normal([channel_in,channel_out], stddev = 0.1), name='W')
            tf.summary.histogram(name + '/Weights', W_f)

        with tf.name_scope('Bias'):
            b_f = tf.Variable(tf.constant(0.1, shape =[channel_out]), name='B')
            tf.summary.histogram(name + '/Bias', b_f)

        with tf.name_scope('fc_layer'):
            output = tf.matmul(input, W_f)+b_f
            tf.summary.histogram(name + '/weights', output)
            return output
## conv1
conv1 = conv_layer(x_image, 1, 32, name='conv1')
## conv2
conv2 = conv_layer(conv1, 32, 64, name = 'conv2')

#func1 layer
with tf.name_scope('func1'):
    h_pool2_flat = tf.reshape(conv2,[-1,7*7*64]) #[n_sampes, 7, 7, 64] ->>> [n_samples, 7*7*64]


h_f1 = tf.nn.relu(fc_layer(h_pool2_flat, 7*7*64, 1024))
h_f1_drop = tf.nn.dropout(h_f1, keep_prob)

#func2 layer
with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(fc_layer(h_f1_drop, 1024, 10))

with tf.name_scope('cross_entropy'):
    cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

sess=tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())

for i in range(train_size):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys:batch_ys , keep_prob: 0.5})

    if i %50 == 0:
        ce = sess.run(merged, feed_dict={xs: batch_xs, ys:batch_ys , keep_prob: 0.5})
        writer.add_summary(ce, i)
        print(i, ' accuracy: ',compute_accuracy(mnist.test.images[:acc_no], mnist.test.labels[:acc_no]))


sess.close()
