#/usr/bin/env python3
#-*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

learning_rate=float(sys.argv[1])
epochs=int(sys.argv[2])

#initiating functions
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d_(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_1x2(x):
    return tf.nn.max_pool(x,ksize=[1,1,2,1],strides=[1,1,2,1],padding='SAME')

def max_pool_1x3(x):
    return tf.nn.max_pool(x,ksize=[1,1,3,1],strides=[1,1,3,1],padding='SAME')

#data preprocessing
data=pd.read_csv('train_se.csv')
data=data.fillna(0)
dataset_x=data
dataset_x=dataset_x.as_matrix()
dataset_x=dataset_x[:,1:-1]
data['ulabel']=data['label'].apply(lambda s:int(not s))
dataset_y=data[['label','ulabel']]
dataset_y=dataset_y.as_matrix()
x_train,x_test,y_train,y_test=train_test_split(dataset_x,dataset_y,test_size=0.2,random_state=42)

#placeholders
x=tf.placeholder(tf.float32,shape=[None,36])
y_=tf.placeholder(tf.float32,shape=[None,2])
x_data=tf.reshape(x,[-1,1,36,1])
keep_prob=tf.placeholder(tf.float32)

#graph
#layer1-convolution
W_conv1=weight_variable([1,3,1,32])
b_conv1=bias_variable([32])
h_conv1=tf.nn.relu(conv2d_(x_data,W_conv1)+b_conv1)
h_pool1=max_pool_1x3(h_conv1)

#layer2-convolution
W_conv2=weight_variable([1,3,32,64])
b_conv2=bias_variable([64])
h_conv2=tf.nn.relu(conv2d_(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_1x3(h_conv2)

#layer3-convolution
W_conv3=weight_variable([1,3,64,128])
b_conv3=bias_variable([128])
h_conv3=tf.nn.relu(conv2d_(h_pool2,W_conv3)+b_conv3)
h_pool3=max_pool_1x2(h_conv2)

#layer4-full connected
W_fc1=weight_variable([256,64])
b_fc1=bias_variable([64])
h_pool3_flat=tf.reshape(h_pool3,[-1,256])
h_fc1=tf.nn.relu(tf.matmul(h_pool3_flat,W_fc1)+b_fc1)

#layer5-dropout
h_drop=tf.nn.dropout(h_fc1,keep_prob)

#layer6-softmax
W_fc2=weight_variable([64,2])
b_fc2=bias_variable([2])
y_conv=tf.nn.softmax(tf.matmul(h_drop,W_fc2)+b_fc2)

#loss
cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv+1e-10),reduction_indices=[1]))

#training settings
train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
acc_step=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#saver
saver=tf.train.Saver()

#train
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for epoch in range(epochs):
        total_loss=0
        for i in range(len(x_train)):
            feed={x:[x_train[i]],y_:[y_train[i]],keep_prob:0.5}
            _,loss=sess.run([train_step,cross_entropy],feed_dict=feed)
            total_loss+=loss
#        print('epoch: %04d, total loss=%.9f' % (epoch+1,total_loss))
    save_path=saver.save(sess,'improse_model.mdl')
#    print('Training complete!')

    #evolution
    accuracy,results = sess.run([acc_step,y_conv], feed_dict={x: dataset_x, y_: dataset_y,keep_prob:1.0})
#    print("Accuracy on validation set: %.9f" % accuracy)

    np.set_printoptions(precision=4,threshold=np.NaN)
    print(results)
