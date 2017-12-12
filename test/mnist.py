import tensorflow as tf
from tensorflow.contrib.keras import layers as klayers
from tensorflow.contrib.keras import preprocessing
ImageDataGenerator = preprocessing.image.ImageDataGenerator
from tensorflow.contrib.keras import datasets
mnist = datasets.mnist
from tensorflow.contrib.keras import utils
to_categorical = utils.to_categorical
# from tensorflow.examples.tutorials.mnist import input_data
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

def get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train = X_train[..., None]
    X_test = X_test[..., None]
    Y_train = to_categorical(y_train, 10)
    Y_test = to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


def get_gen(set_name, batch_size, translate, scale,
            shuffle=True):
    if set_name == 'train':
        (X, Y), _ = get_mnist_dataset()
    elif set_name == 'test':
        _, (X, Y) = get_mnist_dataset()

    image_gen = ImageDataGenerator(
        zoom_range=scale,
        width_shift_range=translate,
        height_shift_range=translate
    )
    gen = image_gen.flow(X, Y, batch_size=batch_size, shuffle=shuffle)
    return gen

def lecun_model(x, winit, nclass=10):
    with sigma.defaults(stride=1, padding='same', act='relu',
                        weight_initializer=winit):
        # 28 x 28 x 1 => 14 x 14 x32
        # tf.summary.image('inputs', x)
        x = layers.convs.conv2d(x, 32, 5)
        # tf.summary.histogram('conv2d_1', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        # tf.summary.histogram('maxpool_1', x)

        # 14 x 14 x 32 => 7 x 7 x 64
        x = layers.convs.conv2d(x, 64, 5)
        # tf.summary.histogram('conv2d_2', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        # tf.summary.histogram('maxpool_2', x)

        # 7 x 7 x 64 =>
        x = layers.base.flatten(x)
        x = layers.convs.dense(x, 1024)
        # tf.summary.histogram('dense_1', x)
        # x = layers.norm.dropout(x, 0.5)
        # tf.summary.histogram('dropout_1', x)
        x = layers.convs.dense(x, nclass, act=None)
        # tf.summary.histogram('dense_2', x)

        return x

def lecun_model_soft(x, winit, nclass=10):
    with sigma.defaults(stride=1, padding='same', act='relu', mode='nearest',
                        weight_initializer=winit):
        # 28 x 28 x 1 => 14 x 14 x32
        # tf.summary.image('inputs', x)
        x = layers.convs.soft_conv2d(x, 32, 5)
        # tf.summary.histogram('conv2d_1', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        # tf.summary.histogram('maxpool_1', x)

        # 14 x 14 x 32 => 7 x 7 x 64
        x = layers.convs.soft_conv2d(x, 64, 5)
        # tf.summary.histogram('conv2d_2', x)
        x = layers.pools.max_pool2d(x, 2, stride=2)
        # tf.summary.histogram('maxpool_2', x)

        # 7 x 7 x 64 =>
        x = layers.base.flatten(x)
        x = layers.convs.dense(x, 1024)
        # tf.summary.histogram('dense_1', x)
        # x = layers.norm.dropout(x, 0.5)
        # tf.summary.histogram('dropout_1', x)
        x = layers.convs.dense(x, nclass, act=None)
        # tf.summary.histogram('dense_2', x)

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

def accuracy(logits, labels):
    y = tf.argmax(tf.nn.softmax(logits), axis=-1)
    _y = tf.argmax(labels, axis=-1)
    acc = tf.cast(tf.reduce_sum(tf.where(tf.equal(y, _y), tf.ones_like(y), tf.zeros_like(y))), dtype=tf.float32)
    return tf.div(acc, tf.cast(tf.shape(y)[0], dtype=tf.float32))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--winit', type=str, default='he_uniform')

    batch_size = 50

    args = parser.parse_args()
    winit = args.winit
    beg = time.time()
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [batch_size, 28, 28, 1], name='samples')
        y = tf.placeholder(tf.float32, [batch_size, 10], name='labels')
        # reshapedx = tf.reshape(x, [-1, 28, 28, 1])

        out = lecun_model_keras(x, winit)
        loss = layers.losses.cce(out, y)
        acc = accuracy(out, y)

        # loss = tf.losses.softmax_cross_entropy(y, out)
        # global_step = tf.Variable(0, trainable=False)
        # decay_lr = tf.train.exponential_decay(0.01, global_step, decay_steps=10000,
        #                                       decay_rate=0.96, staircase=True)
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

        # tf.summary.scalar('loss', loss)

        # mnist = input_data.read_data_sets('test/cache/mnist/', one_hot=True)
        # summarize = tf.summary.merge_all()
        # writer = tf.summary.FileWriter('test/log', sess.graph)
        sess.run(tf.global_variables_initializer())

        losses = []

        train_gen = get_gen(
            'train', batch_size=batch_size,
            scale=(1.0, 2.5), translate=0.2,
            shuffle=True
        )

        test_gen = get_gen(
            'test', batch_size=batch_size,
            scale=(1.0, 2.5), translate=0.2,
            shuffle=False
        )

        for i in range(5000):
            trainx, trainy = next(train_gen)
            _, train_loss = sess.run([train_op, loss],
                                             feed_dict={x:trainx, y:trainy})

            # writer.add_summary(summary, global_step=i)

            validx, validy = next(test_gen)
            valid_loss, _acc = sess.run([loss, acc], feed_dict={x:validx, y:validy})
            # print('train loss: {}, valid loss: {}'.format(train_loss,
            #                                               valid_loss))
            losses.append([valid_loss, _acc])
            # print('epoch: {}, acc: {}'.format(i, _acc))
        # writer.close()
        np.savetxt('test/data/mnist/keras/scaled_{}'.format(winit), losses)
    end = time.time()
    print('soft conv cost: {}'.format(end-beg))
