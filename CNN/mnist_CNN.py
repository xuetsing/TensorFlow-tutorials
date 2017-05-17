# -*- coding:utf-8 -*-

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
#读取图片数据集
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()#创建session

#函数声明部分
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)#正态分布，标准差为0.1，默认最大值为1，最小值为-1，均值为0
    return tf.Variable(initial)
def bias_variable(shape):#创建结构为shape矩阵，初始化所有值为0.1
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x,W):#卷积遍历各方向步数为1，SAME:边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):#池化卷积结果(conv2d),池化层采用kernel大小为2x2，步数为2，周围补0，取最大值。数据量缩小了4倍
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义输入输出结构
xs = tf.placeholder(tf.float32,[None,28*28])#声明一个占位符，None表示输入图片的数量不定，28*28图片分辨率
ys = tf.placeholder(tf.float32,[None,10])#类别是0-9总共10个类别，对应输出分类结果
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs,[-1,28,28,1])#把xs reshape成了28*28*1的形状，因为是灰色图片，所以通道是1，作为训练时的input，-1代表图片数量不定

#搭建网络，定义算法公式
##第一层卷积操作
W_conv1 = weight_variable([5,5,1,32])#第一二参数是卷积核大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像
b_conv1 = bias_variable([32])#每一个卷积核都有一个对应的偏置量
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)#图片乘以卷积核，并加上偏置量，卷积结果28x28x32
h_pool1 = max_pool_2x2(h_conv1)#池化结果14x14x32 卷积结果乘以池化卷积核
##第二层卷积操作
W_conv2 = weight_variable([5,5,32,64])#32通道卷积， 卷积出64个特征
b_conv2 = bias_variable([64])#64个偏置数据
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)#h_pool1是上一层的池化结果，卷积结果14x14x64
h_pool2 = max_pool_2x2(h_conv2)#池化结果7x7x64
#原图像尺寸28x28，第一轮图像缩小为14x14，共有32张，第二轮后图像缩小为7x7，共有64张
##第三层全连接操作
W_fc1 = weight_variable([7*7*64, 1024])#第一个参数7*7*64的patch，也可以认为是只有一行7*7*64个数据的卷积，第二个参数代表卷积个数共1024个
b_fc1 = bias_variable([1024])#1024个偏置数据
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])#将h_pool2结果reshape成只有一行7*7*64个数据，即[n_samples,7,7,64]->>[n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)#卷积操作，结果是1*1*1024,单行乘以单列等于1*1矩阵，matmul实现最基本的矩阵相乘

#dropout操作,减少过拟合
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#第四层输出操作
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
#最后的分类，结果为1*1*10，softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#定义loss(最小误差概率)
cross_entropy = -tf.reduce_sum(ys*tf.log(y_conv))
#调用优化器优化，其实就是通过喂数据争取cross_entropy最小化
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#开始数据训练与评测
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
sess.run(tf.global_variables_initializer())
#tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={xs:batch[0], ys:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={xs: batch[0], ys: batch[1],keep_prob:0.5})
print("test accuracy %g"%accuracy.eval(feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))

















