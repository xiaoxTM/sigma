#! /usr/bin/env python3

import sys
import os
import time
import os.path

import numpy as np

import tensorflow as tf

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../'))
sys.path.append(root)

from dataset import modelnet40_loader
from models.block import _block_conv2d
from sigma import layers, engine, ops, status, helpers
import argparse

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--nclass', default=40, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/cache/checkpoint/model.ckpt', type=str)
parser.add_argument('--log', default='/home/xiaox/studio/exp/3dpcn/cache/log', type=str)
parser.add_argument('--address', default='172.31.234.152:2666')
parser.add_argument('--database', default='modelnet40', type=str)
parser.add_argument('--gpu', default='2', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--learning-rate', default=0.2, type=float)

def build_net(inputs, is_training, nclass=40, reuse=False):
    #inputs: [batch-size, 2048, 3]
    #=>      [batch-size, 3, 2048]
    npoints = ops.core.shape(inputs)[1]
    x = layers.base.transpose(inputs, (0, 2, 1), reuse=reuse, name='transpose-0')
    #        [batch-size, 3, 2048]
    #=>      [batch-size, 6,  512]
    if not reuse:
        ops.core.summarize('inputs', x)
    x = layers.capsules.order_invariance_transform(x, npoints, 9, 'max', reuse=reuse, name='oit', act='squash', weight_regularizer=ops.regularizers.regularize(l1=0.001))
    ##=>    [batch-size, neurons, dims]
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-1')
    x = layers.base.reshape(x, [-1, npoints, 9, 1], reuse=reuse, name='reshape-0')
    x = _block_conv2d(x, 9, kshape=9, reuse=reuse, is_training=is_training, act='relu', name='block_0', reshape=False)
    x = _block_conv2d(x, 9, kshape=9, reuse=reuse, is_training=is_training, act='relu', name='block_1', reshape=False)
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-2')
    x = layers.actives.squash(x, axis=1, epsilon=1e-10, reuse=reuse, safe=True, name='squash')
    ##=>    [batch-size, neurons, dims, 1]
    #x = layers.base.expand_dims(x, 3, reuse=reuse, name='expand_dims')
    #print(ops.core.shape(x))
    #x = layers.capsules.conv1d(x, 3, 32, 3, reuse=reuse, name='cap_conv1d', act='squash')
    #x = layers.capsules.conv1d(x, 6, 16, 3, reuse=reuse, name='cap_conv1d-2', act='squash')
    #x = layers.capsules.conv1d(x, 1, 32, 3, reuse=reuse, name='cap_conv1d-3', act='squash')
    #x = ops.core.squeeze(x, -1)
    #x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-2')
    #x = layers.capsules.order_invariance_transform(x, 256, 20, 'max', reuse=reuse, name='order_invariance_transform-1',
    #        act='tanh')
    #if not reuse:
    #    ops.core.summarize('order_invariance_transform', x)
    #x = layers.capsules.dense(x, 256, 20, reuse=reuse, epsilon=1e-9, name='dense-1', act='squash')
    #if not reuse:
    #    ops.core.summarize('dense-1', x)
    #x = layers.capsules.dense(x, 128, 24, reuse=reuse, epsilon=1e-9, name='dense-2', act='relu')
    #if not reuse:
    #    ops.core.summarize('dense-2', x)
    #x = layers.capsules.dense(x,  64, 48, reuse=reuse, epsilon=1e-9, name='dense-3', act='relu')
    #if not reuse:
    #    ops.core.summarize('dense-3', x)
    #x = layers.capsules.dense(x,  32, 96, reuse=reuse, epsilon=1e-9, name='dense-4', act='relu')
    #if not reuse:
    #    ops.core.summarize('dense-4', x)
    x = layers.capsules.dense(x,  nclass, 24, reuse=reuse, epsilon=1e-10, name='dense-5', act='squash')
    if not reuse:
        ops.core.summarize('dense-5', x)
    x = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-10, name='norm', reuse=reuse)
    if not reuse:
        ops.core.summarize('norm', x)
    return x

@helpers.stampit({'checkpoint':-2, 'log':-1})
def train_net(batch_size=8,
              epochs=1000,
              num_points=2048,
              lr=0.02,
              nclass=40,
              debug=False,
              address=None,
              checkpoint=None,
              log=None,
              database=None,
              gpu='0'):
    engine.set_print(False)

    trainset = modelnet40_loader.ModelNetH5Dataset('/home/xiaox/studio/db/modelnet/ply_hdf5_2018/',
            batch_size=batch_size,
            train=True, npoints=num_points)
    validset = modelnet40_loader.ModelNetH5Dataset('/home/xiaox/studio/db/modelnet/ply_hdf5_2018/',
            batch_size=batch_size,
            train=False, npoints=num_points)

    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)
    inputs = layers.base.input_spec([None, num_points, 3])
    labels = layers.base.label_spec([None, nclass])

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.allow_soft_placement = True
    with ops.core.device('/gpu:0'):
        trainp = build_net(inputs, is_training=True)
        #train_loss_op = layers.losses.get('margin_loss', trainp, labels)
        train_loss_op = layers.losses.categorical_cross_entropy([trainp, labels]) + ops.core.sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        train_metric = layers.metrics.accuracy([trainp, labels])
        train_metric_op, train_metric_update_op, train_metric_initialize_op = train_metric
        learning_rate = tf.train.exponential_decay(lr, global_step, 10000, 0.9)
        update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
        with ops.core.control_dependencies(update_ops):
            train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(train_loss_op, global_step=global_step)

        validp = build_net(inputs, is_training=False, reuse=True)
        #valid_loss_op = layers.losses.get('margin_loss', validp, labels)
        valid_loss_op = layers.losses.get('margin_loss', validp, labels) + ops.core.sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        valid_metric = layers.metrics.accuracy([validp, labels])
        valid_metric_op, valid_metric_update_op, valid_metric_initialize_op = valid_metric
    sess, saver, summarize, writer = engine.session(checkpoint=checkpoint,
                                                    config=config,
                                                    debug=debug,
                                                    address=address,
                                                    log=log)

    #layers.core.export_graph('auto_encoder.png')
    with sess:
        losses =  np.zeros((epochs, 4))
        for epoch in range(epochs):
            start = time.time()
            ops.core.run(sess, train_metric_initialize_op)
            iters = 0
            while trainset.has_next_batch():
                iters += 1
                points, gt = trainset.next_batch(False)
                gt = helpers.one_hot(gt.astype(np.int32), nclass)
                _, loss, _, summary, gstep = ops.core.run(sess, [train_op, train_loss_op, train_metric_update_op, summarize,
                    global_step],
                        feed_dict={inputs:points, labels:gt})
                accuracy = ops.core.run(sess, train_metric_op)
                ops.core.add_summary(writer, summary, global_step=gstep)
                if iters % 10 == 0:
                    print('train for {}-th iteration: loss: {}, accuracy: {}'.format(iters, loss, accuracy))
            trainset.reset()
            end = time.time()
            print('time cost:', end-start)
            # test
            test_loss = []
            test_acc = []
            ops.core.run(sess, valid_metric_initialize_op)
            while validset.has_next_batch():
                points, gt = validset.next_batch()
                gt = helpers.one_hot(gt.astype(np.int32), nclass)
                loss, _ = ops.core.run(sess, [valid_loss_op, valid_metric_update_op], feed_dict={inputs:points,
                    labels:gt})
                accuracy = ops.core.run(sess, valid_metric_op)
                test_loss.append(loss)
                test_acc.append(accuracy)
            validset.reset()
            tloss = np.mean(test_loss)
            tacc = np.mean(test_acc)
            losses[epoch][2] = tloss
            losses[epoch][3] = tacc
            print('test for {}-th epoch: loss:{}, accuracy: {}'.format(epoch, tloss, tacc))
            if epoch % 10 == 0:
                helpers.save(sess, checkpoint, saver, True, global_step=epoch)
        np.savetxt('rec_h5.log', losses)
        ops.core.close_summary_writer(writer)

def eval_net(args):
    inputs, x = build_net(args)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.phase == 'train':
        train_net(args.batch_size,
                  args.epochs,
                  args.num_points,
                  args.learning_rate,
                  args.nclass,
                  args.debug,
                  args.address,
                  args.checkpoint,
                  args.log,
                  args.database,
                  args.gpu)
    elif args.phase == 'eval':
        eval_net(**args)
    else:
        raise ValueError('mode must be either `train` or `eval`. given `{}`'.format(args.phase))
