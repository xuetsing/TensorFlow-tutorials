import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

x_train = mnist.train.images[:55000,:]
#print(x_train.shape)

randomNum = random.randint(0,55000)
image = x_train[randomNum].reshape([28,28])
#plt.imshow(image,cmap=plt.get_cmap('gray_r'))
#plt.show()

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1,1,1,1], padding='SAME')
def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def discriminator(x_image, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if(reuse):
            scope.reuse_variables()
        W_conv1 = tf.get_variable('d_wconv1', [5,5,1,8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = avg_pool_2x2(h_conv1)

        W_conv2 = tf.get_variable('d_wconv2', [5,5,8,16],initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = avg_pool_2x2(h_conv2)

        W_fc1 = tf.get_variable('d_wfc1', [7*7*16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        W_fc2 = tf.get_variable('d_wfc2', [32,1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b_fc2 = tf.get_variable('d_bfc2',[1],initializer=tf.constant_initializer(0))

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        return y_conv

def generator(z, batch_size, z_dim, reuse=False):
    with tf.variable_scope("generator") as scope:
        if(reuse):
            scope.reuse_variables()
        g_dim = 64
        c_dim = 1
        s = 28
        s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8),int(s/16)

        h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
        h0 = tf.nn.relu(h0)

        output1_shape = [batch_size, s8, s8, g_dim*4]
        W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1],int(h0.get_shape()[-1])],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1,2,2,1], padding='SAME')
        H_conv1 = tf.contrib.layers.batch_norm(inputs=H_conv1, center=True, scale=True, is_training=True, scope='g_bn1')
        H_conv1 = tf.nn.relu(H_conv1)

        output2_shape = [batch_size, s4 - 1, s4 - 1, g_dim*2]
        W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
        H_conv2 = tf.nn.relu(H_conv2)

        output3_shape = [batch_size, s2 - 2, s2 - 2, g_dim*1]
        W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME')
        H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
        H_conv3 = tf.nn.relu(H_conv3)

        output4_shape = [batch_size, s, s, c_dim]
        W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
        H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='VALID')
        H_conv4 = tf.nn.tanh(H_conv4)

        return H_conv4

sess = tf.InteractiveSession()
z_dimensions = 100
z_test_placeholder = tf.placeholder(tf.float32,[None, z_dimensions])

sample_image = generator(z_test_placeholder, 1 ,z_dimensions)
test_z = np.random.normal(-1, 1, [1,z_dimensions])

sess.run(tf.global_variables_initializer())
temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))

my_i = temp.squeeze()
print(my_i.shape)
#plt.imshow(my_i, cmap='gray_r')
#plt.show()

# ****************************************************************
batch_size = 16
tf.reset_default_graph()

sess = tf.InteractiveSession()
x_placeholder = tf.placeholder(tf.float32, shape=[None,28,28,1])
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

Dx = discriminator(x_placeholder)
Gz = generator(z_placeholder,batch_size,z_dimensions)
Dg = discriminator(Gz,reuse=True)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(Dg), logits=Dg))

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(Dx), logits=Dx))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(Dg),logits=Dg))
d_loss = d_loss_real + d_loss_fake

tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]


trainerD = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(d_loss, var_list=d_vars)
trainerG = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(g_loss, var_list=g_vars)

sess.run(tf.global_variables_initializer())
iterations = 3000
for i in range(iterations):
    z_batch = np.random.normal(-1,1,size=[batch_size,z_dimensions])
    real_image_batch = mnist.train.next_batch(batch_size)
    real_image_batch = np.reshape(real_image_batch[0],[batch_size,28,28,1])
    _,dLoss = sess.run([trainerD, d_loss],feed_dict={z_placeholder:z_batch,x_placeholder:real_image_batch})
    _,gLoss = sess.run([trainerG, g_loss],feed_dict={z_placeholder:z_batch})
    if i%100 == 0 or (i+1) == iterations:
        print("iter:" + str(i) + "  Gen Loss: " + str(gLoss) + "  Disc Loss: " + str(dLoss))

sample_image = generator(z_placeholder, 1, z_dimensions,reuse=True)
z_batch = np.random.normal(-1,1,size=[1, z_dimensions])
temp = (sess.run(sample_image, feed_dict={z_placeholder:z_batch}))
my_i = temp.squeeze()
print(my_i.shape)
plt.imshow(my_i,cmap='gray_r')
plt.show()



