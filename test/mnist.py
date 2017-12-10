import tensorflow as tf
from tensorflow.contrib.keras import layers as klayers
#import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.examples.tutorials.mnist import input_data
import logging
# logging.basicConfig(level=logging.DEBUG)
# import numpy as np
# np.random.seed(1000)
import os
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
# print(sys.path)
import sigma
from sigma import layers, status, colors
import numpy as np
import argparse

import time

def lecun_model(x, winit, nclass=10):
    with sigma.defaults(stride=1, padding='same', act='relu',
                        weight_initializer=winit):
        # 28 x 28 x 1 => 14 x 14 x32
        # tf.summary.image('inputs', x)
        x = layers.convs.conv2d(x, 32, 5)
        tf.summary.histogram('conv2d_1', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        tf.summary.histogram('maxpool_1', x)

        # 14 x 14 x 32 => 7 x 7 x 64
        x = layers.convs.conv2d(x, 64, 5)
        tf.summary.histogram('conv2d_2', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        tf.summary.histogram('maxpool_2', x)

        # 7 x 7 x 64 =>
        x = layers.base.flatten(x)
        x = layers.convs.dense(x, 1024)
        tf.summary.histogram('dense_1', x)
        # x = layers.norm.dropout(x, 0.5)
        tf.summary.histogram('dropout_1', x)
        x = layers.convs.dense(x, nclass, act=None)
        tf.summary.histogram('dense_2', x)

        return x

def lecun_model_soft(x, winit, nclass=10):
    with sigma.defaults(stride=1, padding='same', act='relu',
                        weight_initializer=winit):
        # 28 x 28 x 1 => 14 x 14 x32
        # tf.summary.image('inputs', x)
        x = layers.convs.soft_conv2d(x, 32, 5)
        tf.summary.histogram('conv2d_1', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        tf.summary.histogram('maxpool_1', x)

        # 14 x 14 x 32 => 7 x 7 x 64
        x = layers.convs.soft_conv2d(x, 64, 5)
        tf.summary.histogram('conv2d_2', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        tf.summary.histogram('maxpool_2', x)

        # 7 x 7 x 64 =>
        x = layers.base.flatten(x)
        x = layers.convs.dense(x, 1024)
        tf.summary.histogram('dense_1', x)
        # x = layers.norm.dropout(x, 0.5)
        tf.summary.histogram('dropout_1', x)
        x = layers.convs.dense(x, nclass, act=None)
        tf.summary.histogram('dense_2', x)

        return x

def lecun_model_keras(x, winit, nclass=10):
    x = klayers.Conv2D(32, (5,5), padding='same', activation='relu',
               kernel_initializer=winit)(x)
    x = klayers.MaxPooling2D()(x)

    x = klayers.Conv2D(64, (5,5), padding='same', activation='relu',
               kernel_initializer=winit)(x)
    x = klayers.MaxPooling2D()(x)

    x = klayers.Flatten()(x)
    x = klayers.Dense(1024, activation='relu', kernel_initializer=winit)(x)

    # x = klayers.Dropout(0.5)(x)

    x = klayers.Dense(10, kernel_initializer=winit)(x)

    return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--winit', type=str, default='he_normal')

    args = parser.parse_args()
    winit = args.winit
    beg = time.time()
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [None, 784], name='samples')
        y = tf.placeholder(tf.float32, [None, 10], name='labels')
        reshapedx = tf.reshape(x, [-1, 28, 28, 1])

        out = lecun_model_soft(reshapedx, winit)
        loss = layers.losses.cce(out, y)

        # loss = tf.losses.softmax_cross_entropy(y, out)
        # global_step = tf.Variable(0, trainable=False)
        # decay_lr = tf.train.exponential_decay(0.01, global_step, decay_steps=10000,
        #                                       decay_rate=0.96, staircase=True)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

        # tf.summary.scalar('loss', loss)

        mnist = input_data.read_data_sets('test/cache/mnist/', one_hot=True)
        # summarize = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('test/log', sess.graph)
        sess.run(tf.global_variables_initializer())

        losses = []

        for i in range(100):
            trainx, trainy = mnist.train.next_batch(50)
            _, train_loss = sess.run([train_op, loss],
                                             feed_dict={x:trainx, y:trainy})
            losses.append(train_loss)

            # writer.add_summary(summary, global_step=i)

            validx, validy = mnist.test.next_batch(50)
            valid_loss = sess.run(loss, feed_dict={x:validx, y:validy})
            print('train loss: {}, valid loss: {}'.format(train_loss,
                                                          valid_loss))
        # writer.close()
        np.savetxt('test/data/debug/soft_{}'.format(winit), losses)
    end = time.time()
    print('soft conv cost: {}'.format(end-beg))
