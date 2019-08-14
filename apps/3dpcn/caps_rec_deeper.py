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
from sigma import layers, engine, ops, status, helpers
import argparse

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--nclass', default=40, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/cache/checkpoint/model.ckpt', type=str)
parser.add_argument('--log', default='/home/xiaox/studio/exp/3dpcn/cache/log', type=str)
#parser.add_argument('--address', default='172.31.234.152:2666')
parser.add_argument('--address', default=None)
parser.add_argument('--database', default='modelnet', type=str)
parser.add_argument('--gpu', default='3', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--learning-rate', default=0.8, type=float)

def projection_transform(inputs,
                         dims,
                         weight_initializer='glorot_uniform',
                         weight_regularizer=None,
                         bias_initializer='zeros',
                         bias_regularizer=None,
                         cpuid=0,
                         act=None,
                         trainable=True,
                         dtype=ops.core.float32,
                         collections=None,
                         summary='histogram',
                         reuse=False,
                         name=None,
                         scope=None):
    #helper.check_input_shape(input_shape)
    input_shape = ops.core.shape(inputs)
    if ops.helper.is_tensor(input_shape):
        input_shape = input_shape.as_list()
    if len(input_shape) != 3:
        raise ValueError('fully_conv require input shape {}[batch-size,'
                         ' dims, channels]{}, given {}'
                         .format(colors.fg.green, colors.reset,
                                 colors.red(input_shape)))
    batch_size, indims, incaps = input_shape
    weight_shape = [1, indims, dims, incaps] # get rid of batch_size axis
    bias_shape = [incaps]
    output_shape = [input_shape[0], dims, incaps]
    ops_scope, _, name = ops.helper.assign_scope(name,
                                             scope,
                                             'projection_transform',
                                             reuse)
    act = ops.actives.get(act)
    weights = ops.mm.malloc('weights',
                        name,
                        weight_shape,
                        dtype,
                        weight_initializer,
                        weight_regularizer,
                        cpuid,
                        trainable,
                        collections,
                        summary,
                        reuse,
                        scope)
    if not isinstance(bias_initializer, bool) or bias_initializer is True:
        bias = ops.mm.malloc('bias',
                         name,
                         bias_shape,
                         dtype,
                         bias_initializer,
                         bias_regularizer,
                         cpuid,
                         trainable,
                         collections,
                         summary,
                         reuse,
                         scope)
    else:
        bias = 0
    def _projection_transform(x):
        with ops_scope:
            #    [batch-size, indims, incaps]
            #=>  [batch-size, indims, 1, incaps]
            x = ops.core.expand_dims(x, 2)
            #    [batch-size, indims, 1, incaps]
            #  * [1, indims, dims, 1]
            #=>  [batch-size, indims, dims, incaps] (*)
            #=>  [batch-size, dims, incaps] (sum)
            x = ops.core.sum(x * weights, axis=1) + bias
            return act(x)

    return _projection_transform(inputs)

def build_net(inputs, nclass=16, reuse=False, is_training=True):
    #inputs: [batch-size, npoints, 3]
    #=>      [batch-size, 3, npoints]
    x = layers.base.transpose(inputs, (0, 2, 1))
    #        [batch-size, 3, npoints]
    if not reuse:
        ops.core.summarize('inputs', x)
    #=> [batch-size, 64, npoints]
    x = projection_transform(x, 9, reuse=reuse, name='projt-0', act='squash')
    if not reuse:
        ops.core.summarize('projt-0', x)
    #x = layers.norms.batch_norm(x, is_training, reuse=reuse, name='batchnorm-0')
    #if not reuse:
    #    ops.core.summarize('batchnorm-0', x)
    #x = layers.actives.relu(x, reuse=reuse, name='relu-0')
    #if not reuse:
    #    ops.core.summarize('relu-0', x)
    #=> [batch-size, 64, npoints]
    x = projection_transform(x, 9, reuse=reuse, name='projt-1', act='squash')
    if not reuse:
        ops.core.summarize('projt-1', x)
    #x = layers.norms.batch_norm(x, is_training, reuse=reuse, name='batchnorm-1')
    #if not reuse:
    #    ops.core.summarize('batchnorm-1', x)
    #x = layers.actives.relu(x, reuse=reuse, name='relu-1')
    #if not reuse:
    #    ops.core.summarize('relu-1', x)
    x = layers.capsules.order_invariance_transform(x, 24, 16, 'max', reuse=reuse, name='projt-2', act='squash')
    if not reuse:
        ops.core.summarize('projt-2', x)
    #x = layers.actives.squash(x, axis=1, reuse=reuse, name='squash-1')
    #if not reuse:
    #    ops.core.summarize('squash', x)
    x = layers.capsules.dense(x, nclass, 16, reuse=reuse, epsilon=1e-9, name='dense-5', act='squash')
    if not reuse:
        ops.core.summarize('dense-5', x)
    x = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-9, name='norm', reuse=reuse)
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
            batch_size=batch_size, train=True, npoints=num_points)
    validset = modelnet40_loader.ModelNetH5Dataset('/home/xiaox/studio/db/modelnet/ply_hdf5_2018/',
            batch_size=batch_size, train=False, npoints=num_points)

    inputs = layers.base.input_spec([None, num_points, 3])
    labels = layers.base.label_spec([None, nclass])

    #ops.core.summarize('inputs', inputs)
    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.allow_soft_placement = True
    with ops.core.device('/gpu:0'):
        trainp = build_net(inputs, nclass)
        train_loss_op = layers.losses.get('margin_loss', trainp, labels)
        train_metric = layers.metrics.accuracy([trainp, labels])
        train_metric_op, train_metric_update_op, train_metric_initialize_op = train_metric
        learning_rate = tf.train.exponential_decay(lr, global_step, 9843, 0.9)
        update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
        with ops.core.control_dependencies(update_ops):
            train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(train_loss_op, global_step=global_step)

        validp = build_net(inputs, nclass, reuse=True, is_training=False)
        valid_loss_op = layers.losses.get('margin_loss', validp, labels)
        valid_metric = layers.metrics.accuracy([validp, labels])
        valid_metric_op, valid_metric_update_op, valid_metric_initialize_op = valid_metric
    ops.core.summarize('trainloss', train_loss_op, 'scalar')
    ops.core.summarize('trainacc', train_metric_op, 'scalar')
    #ops.core.summarize('validloss', valid_loss_op, 'scalar')
    #ops.core.summarize('validacc', valid_metric_op, 'scalar')
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
                points, gt = trainset.next_batch()
                gt = helpers.one_hot(gt.astype(np.int32), nclass)
                #_, loss, _,  gstep = ops.core.run(sess, [train_op, train_loss_op, train_metric_update_op, global_step], feed_dict={inputs:points, labels:gt})
                _, loss, _, summary, gstep = ops.core.run(sess, [train_op, train_loss_op, train_metric_update_op,
                    summarize, global_step], feed_dict={inputs:points, labels:gt})
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
        np.savetxt('losses.log', losses)
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
