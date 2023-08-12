import numpy as np
import scipy.signal
import librosa
import tensorflow as tf
from UNET_model import UNET
from config import *
from GetData_MakeSpec import get_data,MakeSpec
import os

X = tf.placeholder(tf.float32, [None, 1024, 256, 1])
Y = tf.placeholder(tf.float32, [None, 1024, 256, 4])
train_mode = tf.placeholder(tf.bool)  # Feed True or False
global_step = tf.Variable(0, trainable=False, name='global_step')
# TRAIN

unet = UNET(X, train_mode)

X_multi_channel = tf.concat((X, X, X, X), axis=-1)
cost = tf.reduce_sum(tf.abs(tf.subtract(tf.multiply(unet, X_multi_channel), Y)))
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)


with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    train_patch_list = os.listdir(r'E:\Stem_Np\Train_frame')
    for epoch in range(1):
        for i in range(1):  # range(EPOCH), for j in range(num of files)
            # a, b = MakeSpec()
            # d1, d2 = MakeSpec2()
            a = np.load(r'E:\Stem_Np\Train_patch\X\patch_0.npy')
            b = np.load(r'E:\Stem_Np\Train_patch\Y\patch_0.npy')
            print(a.shape)
            _, c = sess.run([optimizer, cost], feed_dict={X: a, Y: b, train_mode: True})
            # print(type(c))
            print(epoch, i+1, c)
        print(type(cost))
    # save weights
    # c_test = sess.run(cost, feed_dict={X: a, Y: b, train_mode: False})

    saver.save(sess, './ckpt/unet.ckpt', global_step=global_step)
    print(c)
    # test_patch_list = os.listdir('/workspace/seungho/Data/Test_patch/X')
    # for test_num in range(len(test_patch_list)):
    #     sess.run(cost, feed_dict={X: a, Y: b, train_mode: False})
    #     pass
# print(c_test)
