import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, dbs, ops, engine, colors, helpers
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tfdbg
import logging
import os

import time

from tensorflow.examples.tutorials.mnist import input_data

#import skimage as sk
#import skimage.io as skio

logging.basicConfig(level=logging.INFO)

def build_func(inputs, labels, initializer='glorot_uniform'):
    # inputs shape :
    #    [batch-size, 28x28]
    ops.core.summarize('inputs', inputs)
    image = layers.base.reshape(inputs, [-1, 28, 28, 1])
    x = layers.convs.conv2d(image, 256, 9,
                            stride=1,
                            padding='valid',
                            act='relu',
                            name='conv2d-0',
                            weight_initializer=initializer)
    ops.core.summarize('conv2d-0', x)
    #   [batch-size, 20, 20, 256]
    #=> [batch-size, 6, 6, 256]
    x = layers.convs.conv2d(x, 32*8, 9,
                            stride=2,
                            act='relu',
                            padding='valid',
                            name='conv2d-1',
                            weight_initializer=initializer)
    ops.core.summarize('conv2d-1', x)
    #   [batch-size, 6, 6, 256]
    #=> [batch-size, 6 * 6 * 32, 8]
    #=> [batch-size, 8, 6 * 6 * 32]
    x = layers.base.reshape(x, [-1, 6*6*32, 8])
    x = layers.base.transpose(x, (0, 2, 1))
    x = layers.actives.squash(x, axis=1, epsilon=1e-9, name='squash-actives')
    ops.core.summarize('squash-actives', x)

    #  [batch_size, nrows * ncols * 32, 8]
    #=>[batch_size, 16, 10]
    # a.k.a [batch-size, 1=capsules of each channel, 16=capsule atoms/dims, 10=channels]
    # digitCapsule Layer
    random_normal = ops.initializers.get('random_normal', stddev=0.01)
    x = layers.capsules.dense(x, 10, 16, 3,
                              share_weights=False,
                              name='capdense-0',
                              safe=True,
                              epsilon=1e-9,
                              weight_initializer=random_normal)
    ops.core.summarize('capdense-0', x)
    # norm the output to represent the existance probabilities
    # of each class
    # classification:
    #    [batch-size, caps_dims, incaps=channels]
    #=>  [batch-size, incaps=channels=10]
    classification = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-9)
    ops.core.summarize('norm-0', classification)
    class_loss = layers.losses.get('margin_loss', classification, labels)
    #loss = class_loss
    tf.summary.scalar('classification-loss', class_loss)
    ## reconstruction
    x = layers.base.maskout(x, index=labels)
    ops.core.summarize('maskout', x)
    x = layers.convs.dense(x, 512, act='relu', name='dense-0')
    ops.core.summarize('dense-0', x)
    x = layers.convs.dense(x, 1024, act='relu', name='dense-1')
    ops.core.summarize('dense-1', x)
    x = layers.convs.dense(x, 784, act='sigmoid', name='dense-2')
    ops.core.summarize('dense-2', x)
    reconstruction = layers.base.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('reconstruction', reconstruction, max_outputs=10)
    recon_loss = layers.losses.mse([reconstruction, image])
    tf.summary.scalar('reconstruction-loss', recon_loss)
    loss = layers.math.add([class_loss, recon_loss], [1, 0.005])
    metric = layers.metrics.accuracy([classification, labels])
    ops.core.summarize('loss', loss, 'scalar')
    ops.core.summarize('acc', metric[0], 'scalar')
    return reconstruction, loss, metric
    #return None, loss, metric

def train(epochs=100, batchsize=100, checkpoint=None, logdir=None):
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

    with ops.core.device('/gpu:0'):
        reconstruction_op, loss_op, metrics_op = build_func(inputs, labels)
        metric_op, metric_update_op, metric_variable_initialize_op = metrics_op
        learning_rate = tf.train.exponential_decay(0.001, global_step, mnist.train.num_examples / batchsize, 0.998)
        train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(loss_op, global_step=global_step)
        #trainable_variables = tf.trainable_variables()
        #grads = tf.gradient(loss_op, trainables)
        #optimizer = ops.optimizers.get('AdamOptimizer')
        #train_op = optimizer.apply_gradients(zip(grad, trainable_variables))

    sess, saver, summarize, writer = engine.session(config=config,
                                                    checkpoint=checkpoint,
                                                    debug=False,
                                                    #address='172.31.234.152:2666',
                                                    log=logdir)
    #base = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing/mnist'
    #with sess:
    #    #tf.global_variables_initializer().run()
    #    #tf.local_variables_initializer().run()

    #    mlog = np.ones([epochs, 5])

    #    steps = int(mnist.train.num_examples / batchsize)
    #    for epoch in range(epochs):
    #        start = time.time()
    #        ops.core.run(sess, metric_variable_initialize_op)
    #        for step in range(steps):
    #            xs, ys = mnist.train.next_batch(batchsize, shuffle=True)
    #            train_feed = {inputs: xs, labels: ys}
    #            _, loss, _, summary = ops.core.run(sess, [train_op, loss_op, metric_update_op, summarize], feed_dict=train_feed)
    #            #loss, _, summary = sess.run([loss_op, metric_update_op, summarize], feed_dict=train_feed)
    #            metric = sess.run(metric_op)
    #            ops.core.add_summary(writer, summary, global_step=(epoch * steps) + step)
    #            if step % 20 == 0:
    #                print('train for {}-th iteration: loss:{}, accuracy:{}'.format(step, loss, metric))
    #        end = time.time()
    #        mlog[epoch][4] = end - start
    #        print("time cost:", mlog[epoch][4])
    #        valid_step = int(mnist.validation.num_examples / batchsize)
    #        validation_loss = []
    #        validation_metric = []
    #        ops.core.run(sess, metric_variable_initialize_op)
    #        for vstep in range(valid_step):
    #            xs, ys = mnist.validation.next_batch(batchsize)
    #            valid_feed = {inputs:xs, labels:ys}
    #            loss, _ = sess.run([loss_op, metric_update_op], feed_dict=valid_feed)
    #            metric = ops.core.run(sess, metric_op)
    #            validation_loss.append(loss)
    #            validation_metric.append(metric)
    #        vloss = np.asarray(validation_loss).mean()
    #        vmetric = np.asarray(validation_metric).mean()
    #        mlog[epoch][0] = vloss
    #        mlog[epoch][1] = vmetric
    #        print('valid for {}-th epoch: loss:{}, accuracy:{}'.format(epoch, vloss, vmetric))

    #        test_step = int(mnist.test.num_examples / batchsize)
    #        test_loss = []
    #        test_metric = []
    #        ops.core.run(sess, metric_variable_initialize_op)
    #        for tstep in range(test_step):
    #            xs, ys = mnist.test.next_batch(batchsize)
    #            test_feed = {inputs:xs, labels:ys}
    #            loss, _ = sess.run([loss_op, metric_update_op], feed_dict=test_feed)
    #            reconstruction, loss, _ = sess.run([reconstruction_op, loss_op, metric_update_op], feed_dict=test_feed)
    #            metric = ops.core.run(sess, metric_op)
    #            test_loss.append(loss)
    #            test_metric.append(metric)
    #            if epoch % 10 == 0:
    #                for idx, (predict, origin) in enumerate(zip(reconstruction, xs)):
    #                    origin = sk.img_as_ubyte(np.reshape(origin, [28, 28, 1]))
    #                    predict = sk.img_as_ubyte(predict)
    #                    #print(origin.shape, predict.shape)
    #                    os.makedirs('{}/{}/{}'.format(base, epoch, tstep), exist_ok=True)
    #                    images = np.concatenate([origin, predict], axis=1)
    #                    skio.imsave('{}/{}/{}/{}.png'.format(base, epoch, tstep, idx), images)
    #        tloss = np.asarray(test_loss).mean()
    #        tmetric = np.asarray(test_metric).mean()
    #        mlog[epoch][2] = tloss
    #        mlog[epoch][3] = tmetric
    #        print('test for {}-th epoch: loss:{}, accuracy:{}'.format(epoch, tloss, tmetric))
    #        if epoch % 10 == 0:
    #            helpers.save(sess, checkpoint, saver, True)
    #    ops.core.close_summary_writer(writer)
    #    np.savetxt('log', mlog)


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', '-c', type=str, default=None)
parser.add_argument('--log', '-l', type=str, default=None)
parser.add_argument('--timestamp', '-t', type=bool, default=True)
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--batch-size', '-b', type=int, default=100)

if __name__=='__main__':
    args = parser.parse_args()
    exp = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing'
    if args.timestamp:
        exp = os.path.join(exp, helpers.timestamp(fmt='%Y%m%d%H%M%S', split=None))
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    checkpoint = None
    log = None
    if args.checkpoint is not None:
        os.makedirs('{}/{}'.format(exp, args.checkpoint), exist_ok=True)
        checkpoint = os.path.join(exp, args.checkpoint, 'ckpt')
    if args.log is not None:
        os.makedirs('{}/{}'.format(exp, args.log), exist_ok=True)
        log = os.path.join(exp, args.log)
    train(args.epochs, args.batch_size, checkpoint=checkpoint, logdir=log)
