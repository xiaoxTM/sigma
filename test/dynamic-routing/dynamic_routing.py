import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, dbs, ops, engine, colors, helpers
import os.path
import numpy as np
import tensorflow as tf
import logging
import os

import time

from tensorflow.examples.tutorials.mnist import input_data

import skimage as sk
import skimage.io as skio

logging.basicConfig(level=logging.INFO)

def build_func(inputs, labels, initializer='glorot_normal', pdim=8, ddim=16, share_weights=False):
    # inputs shape :
    #    [batch-size, 28x28]
    # ops.core.summarize('inputs', inputs)
    image = layers.base.reshape(inputs, [-1, 28, 28, 1])
    x = layers.convs.conv2d(image, 256, 9,
                            padding='valid',
                            act='relu',
                            weight_initializer=initializer)
    # ops.core.summarize('conv2d-0', x)
    #  [batch-size, rows, cols, 256]
    #=>[batch-size, rows, cols, 1, 256]
    x = layers.base.expand_dims(x, -2)
    # no routing between conv1 and primary caps
    # routing means:
    #     `reaching a agreement for all capsules in lower layer`
    # since the lower contains ONLY one capsule
    # we disable routing by setting iterations to 1
    # x shape:
    #  [batch-size, nrows, ncols, 32, 8]
    # primaryCapsule Layer
    x, outshape = layers.capsules.conv2d(x, 32, pdim, 9, 1,
                                         share_weights=share_weights,
                                         stride=2,
                                         # activation for pre-predictions
                                         # that is, u^{hat}_{j|i}
                                         #act='leaky_relu',
                                         safe=True,
                                         weight_initializer=initializer,
                                         return_shape=True)
    # ops.core.summarize('conv2d-1', x)
    # x, outshape = layers.capsules.conv2d(x, 64, 32, 5, 1,
    #                                      stride=2,
    #                                      weight_initializer=initializer,
    #                                      return_shape=True)
    # ops.core.summarize('conv2d-2', x)

    #  [batch-size, nrows, ncols, 32, 8]
    #+>[batch_size, nrows * ncols * 32, 8]
    x = layers.base.reshape(x,
                            [outshape[0],
                             np.prod(outshape[1:-1]),
                             outshape[-1]])
    #  [batch_size, nrows * ncols * 32, 8]
    #=>[batch_size, 10, 16]
    # digitCapsule Layer
    x = layers.capsules.dense(x, 10, ddim, 3, share_weights=share_weights,
                              weight_initializer=initializer)
    # ops.core.summarize('fully_connected-0', x)
    # norm the output to represent the existance probabilities
    # of each class
    classification = layers.capsules.norm(x)
    # ops.core.summarize('norm-0', classification)
    class_loss = layers.losses.get('margin_loss', classification, labels)
    #loss = class_loss
    #tf.summary.scalar('classification-loss', class_loss)
    ## reconstruction
    x = layers.base.maskout(x, index=labels)
    x = layers.convs.dense(x, 512, act='relu')
    x = layers.convs.dense(x, 1024, act='relu')
    x = layers.convs.dense(x, 784, act='sigmoid')
    reconstruction = layers.base.reshape(x, [-1, 28, 28, 1])
    #tf.summary.image('reconstruction', reconstruction, max_outputs=10)
    recon_loss = layers.losses.mse([reconstruction, image])
    #tf.summary.scalar('reconstruction-loss', recon_loss)
    loss = layers.math.add([class_loss, recon_loss], [1, 0.005])
    metric = layers.metrics.accuracy([classification, labels])
    # ops.core.summarize('loss', loss, 'scalar')
    # ops.core.summarize('acc', metric[0], 'scalar')
    return reconstruction, loss, metric

def train(epochs=100, batchsize=100, checkpoint=None, mode=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    # config.gpu_options.visible_device_list = '0,1,2,3'
    # config.intra_op_parallelism_threads = 1
    ## allow TensorFlow put operations on CPU if operations
    ## can not been put on GPU
    config.allow_soft_placement = True

    filefolds = '/home/xiaox/studio/db/mnist'
    mnist = input_data.read_data_sets(filefolds, one_hot=True)

    inputs = layers.base.input_spec([None, 784])
    labels = layers.base.label_spec([None, 10])
    global_step = tf.Variable(0, trainable=False)

    share_weights = False
    if mode == 'share-weights':
        share_weights = True

    with ops.core.device('/gpu:0'):
        reconstruction_op, loss_op, metrics_op = build_func(inputs, labels, share_weights=share_weights)
        metric_op, metric_update_op, metric_variable_initialize_op = metrics_op
        learning_rate = tf.train.exponential_decay(0.01, global_step, mnist.train.num_examples / batchsize, 0.998)
        train_op = ops.optimizers.get('AdamOptimizer').minimize(loss_op, global_step=global_step)

    sess = tf.Session(config=config)
    if checkpoint is not None:
        sess, saver = helpers.load(sess, checkpoint, verbose=True)

    base = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing/mnist/{}'.format(mode)
    with sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        mlog = np.ones([epochs, 5])

        for epoch in range(epochs):
            start = time.time()
            steps = int(mnist.train.num_examples / batchsize)
            ops.core.run(sess, metric_variable_initialize_op)
            for step in range(steps):
                xs, ys = mnist.train.next_batch(batchsize, shuffle=True)
                train_feed = {inputs: xs, labels: ys}
                _, loss, _ = sess.run([train_op, loss_op, metric_update_op], feed_dict=train_feed)
                metric = sess.run(metric_op)
                if step % 20 == 0:
                    print('train for {}-th iteration: loss:{}, accuracy:{}'.format(step, loss, metric))
            end = time.time()
            mlog[epoch][4] = end - start
            #print("time cost:", mlog[epoch][4])
            valid_step = int(mnist.validation.num_examples / batchsize)
            validation_loss = []
            validation_metric = []
            ops.core.run(sess, metric_variable_initialize_op)
            for vstep in range(valid_step):
                xs, ys = mnist.validation.next_batch(batchsize)
                valid_feed = {inputs:xs, labels:ys}
                loss, _ = sess.run([loss_op, metric_update_op], feed_dict=valid_feed)
                metric = ops.core.run(sess, metric_op)
                validation_loss.append(loss)
                validation_metric.append(metric)
            vloss = np.asarray(validation_loss).mean()
            vmetric = np.asarray(validation_metric).mean()
            mlog[epoch][0] = vloss
            mlog[epoch][1] = vmetric
            print('valid for {}-th epoch: loss:{}, accuracy:{}'.format(epoch, vloss, vmetric))

            test_step = int(mnist.test.num_examples / batchsize)
            test_loss = []
            test_metric = []
            ops.core.run(sess, metric_variable_initialize_op)
            for tstep in range(test_step):
                xs, ys = mnist.test.next_batch(batchsize)
                test_feed = {inputs:xs, labels:ys}
                reconstruction, loss, _ = sess.run([reconstruction_op, loss_op, metric_update_op], feed_dict=test_feed)
                metric = ops.core.run(sess, metric_op)
                test_loss.append(loss)
                test_metric.append(metric)
                if epoch % 10 == 0:
                    for idx, (predict, origin) in enumerate(zip(reconstruction, xs)):
                        origin = sk.img_as_ubyte(np.reshape(origin, [28, 28, 1]))
                        predict = sk.img_as_ubyte(predict)
                        #print(origin.shape, predict.shape)
                        os.makedirs('{}/{}/{}'.format(base, epoch, tstep), exist_ok=True)
                        images = np.concatenate([origin, predict], axis=1)
                        skio.imsave('{}/{}/{}/{}.png'.format(base, epoch, tstep, idx), images)
            tloss = np.asarray(test_loss).mean()
            tmetric = np.asarray(test_metric).mean()
            mlog[epoch][2] = tloss
            mlog[epoch][3] = tmetric
            print('test for {}-th epoch: loss:{}, accuracy:{}'.format(epoch, tloss, tmetric))
            if epoch % 10 == 0:
                helpers.save(sess, checkpoint, saver, True)
    os.makedirs(mode, exist_ok=True)
    np.savetxt('{}/log'.format(mode), mlog)


if __name__=='__main__':
    exp = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    #mode = 'non-share-weights' # 'non-share-weights', 'share-weights'
    mode = sys.argv[1]
    os.makedirs('{}/{}/cache/checkpoint'.format(exp, mode), exist_ok=True)
    checkpoint = os.path.join(exp, mode, 'cache/checkpoint/ckpt')
    train(checkpoint=checkpoint, mode=mode)
