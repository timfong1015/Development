#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 10:45:58 2017

@author: Tim
Based on code at https://www.tensorflow.org/get_started/mnist/pros
Modified with tensorboard output
THINK ABOUT SETTING THIS UP TO RUN NON INTERACTIVELY
"""
#get dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
 
import tensorflow as tf

logs_path = "/Users/Tim/Development/logs_DeepMNIST"

#interactive session here
sess = tf.InteractiveSession()

#placeholders
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

def variable_summaries(var):
        #attach a  lot of summaries to a tensor for visualization. From
        #https://www.tensorflow.org/get_started/summaries_and_tensorboard
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))    
            tf.summary.scalar('min', tf.reduce_min(var))
            #tf.summary.historgram('histogram', var)

#weight and bias definitions for the layers
def weight_variable(shape):
    with tf.name_scope('weight'):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

def bias_variable(shape):
    with tf.name_scope('bias'):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
 
#Convolution and pooling definitions
def conv2d(x, W):
    with tf.name_scope('conv_layer'):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    with tf.name_scope('max_pool_2x2'):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
        
#FIRST CONV LAYER
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

#reLU and pooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1)


#SECOND CONV LAYER
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

#reLU and pooling 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#FULLY CONNECTED LAYER
with tf.name_scope('FC_Layer'):
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])

#Flattening and reLU
with tf.name_scope('Flatten_Layer'):
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#DROPOUT
with tf.name_scope('Dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#READOUT LAYER
with tf.name_scope('Readout_Layer'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#TRAIN MODEL
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope('optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

#Evaluate model
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Create cost and accuracy summaries and merge Tensorboard summaries
tf.summary.scalar("cost", cost)
tf.summary.scalar("accuracy", accuracy)
summary_op = tf.summary.merge_all()

#Run model
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(logs_path, sess.graph)

for i in range(100):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
         
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuaracy %g"%accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


    
    
        
        

