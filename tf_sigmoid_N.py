import tensorflow as tf 
import numpy as np
def add_layer(input, in_size, out_size, activation_function=None):
	weight = tf.Variable(tf.random_normal([in_size, out_size]))
	bias = tf.Variable(tf.zeros([1,out_size])+0.1)
	Wx_b = tf.matmul(input, weight) + bias
	if activation_function is None:
		output = Wx_b
	else:
		output = activation_function(Wx_b)
	return output



X = np.array([[1,0,1],[0,1,1],[0,0,1]])
Y = np.array([[1,1,0]]).T


x_data = np.asarray(X)
y_data = np.asarray(Y)

xs = tf.placeholder(tf.float32, x_data.shape)
ys = tf.placeholder(tf.float32, y_data.shape)

l1 = add_layer(xs, 3, 4, activation_function = tf.nn.relu)

prediction = add_layer(l1, 4, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess =tf.Session()
sess.run(init)

for i in range(1000):
	sess.run(train_step, feed_dict={xs: x_data, ys: y_data})

	if i% 50 == 0:
		print(i, sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
		#print(sess.run(l1, feed_dict={xs:x_data, ys:y_data}))