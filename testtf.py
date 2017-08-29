import numpy as np
import tensorflow as tf

num_dims = 2
num_samples = 10
wtarget = np.random.random((num_dims, num_dims)).astype(np.float32)
winit = np.random.random((num_dims, num_dims)).astype(np.float32)
x = np.random.random((num_samples, num_dims)).astype(np.float32)
wx = np.dot(x, wtarget).astype(np.float32)
#print w, '\n', x, '\n', wx

input = tf.constant(x, name='input')
y_ = tf.constant(wx, name='y_')
W = tf.Variable(tf.zeros([num_dims, num_dims]), name='W')
b = tf.Variable(tf.zeros([num_dims]), name='b')

y = tf.matmul(input, W) + b
diff = (y - y_) **2
sums =  tf.reduce_sum(diff, [1])
loss = tf.reduce_mean(sums)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(300):
        # print 'input:', sess.run(input)
        # print 'y_:', sess.run(y_)
        # print 'b:', sess.run(b)
        # print 'W:', sess.run(W)
        # print 'y:', sess.run(y)
        # print 'diff:', sess.run(diff)
        # print 'sums:', sess.run(sums)
        sess.run(train_step)
        print 'loss:', sess.run(loss)

