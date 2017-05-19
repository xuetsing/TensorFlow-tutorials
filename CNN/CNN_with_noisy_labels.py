# -*- coding:utf-8 -*-

import tensorflow as tf
import random
import numpy as np
import datetime
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial = tf.constant(0.1,shape=shape)
	return tf.Variable(initial)
def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x,[-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
MSE = tf.reduce_sum(tf.square(y_ - y_conv))

train_step = tf.train.AdamOptimizer().minimize(MSE)

correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#sess.run(tf.initialize_all_variables())
sess.run(tf.global_variables_initializer())

changeLabelProbability = 0.5
tf.summary.scalar('MSE_Loss',MSE)
tf.summary.scalar('Accuracy',accuracy)
merged = tf.summary.merge_all()
logdir = "log/" + "ChangeLabelProb:" + str(changeLabelProbability) + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

batchSize = 50
for i in range(2000):
    batch = mnist.train.next_batch(batchSize)
    if i%10 == 0:
        summary = sess.run(merged, {x: batch[0], y_: batch[1], keep_prob: 1.0})
        writer.add_summary(summary, i)
    if i%100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
        print ("step %d, training accuracy %g"%(i, train_accuracy))
    # NOISY LABELS
    if (random.random() < changeLabelProbability):
        for i in range(batchSize):
            originalTrainingLabel = np.argmax(batch[1][i])
            validOtherChoices = list(range(0,originalTrainingLabel)) + list(range(originalTrainingLabel+1,10))
            newTrainingLabel = random.choice(validOtherChoices)
            batch[1][i] = np.zeros(10)
            batch[1][i][newTrainingLabel] = 1
    train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.2})

for i in range(10):
    batch = mnist.test.next_batch(batchSize)
    print(sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1}))
print("test accuracy: %g" %accuracy.eval(session=sess, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob: 1.0}))

