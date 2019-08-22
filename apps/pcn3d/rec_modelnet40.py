#! /usr/bin/env python3

import sys
import os
import time
import os.path

import numpy as np

import tensorflow as tf

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../../'))
sys.path.append(root)

from sigma import layers, engine, ops, status, helpers
import argparse

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=2, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--nclass', default=40, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/cache/checkpoint/model.ckpt', type=str)
parser.add_argument('--log', default='/home/xiaox/studio/exp/3dpcn/cache/log', type=str)
parser.add_argument('--address', default='172.31.234.152:2666')
parser.add_argument('--database', default='modelnet40', type=str)
parser.add_argument('--gpu', default='3', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--learning-rate', default=0.05, type=float)

def point_capsule_tio(inputs,
                      is_training,
                      nclass=40,
                      reuse=False):
    #    [batch-size, npoints, 3]
    npoints = ops.core.shape(inputs)[1]
    #    [batch-size, npoints, 3]
    #=>  [batch-size, 3, npoints]; `npoints` capsules, each of which has 3 dims
    x = layers.base.transpose(inputs, (0, 2, 1), reuse=reuse, name='transpose-0')
    x = layers.capsules.order_invariance_transform(x, npoints, 9, act='squash', name='tio', reuse=reuse)
    #    [batch-size, 18, npoints]
    #=>  [batch-size, npoints, 18]
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-1')
    #    [batch-size, npoints, 18, 1]
    x = layers.base.reshape(x, [-1, npoints, 9, 1], name='reshape', reuse=reuse)
    #    [batch-size, npoints, 3, 16]
    x = _block_conv2d(x, 16, kshape=9, reuse=reuse, is_training=is_training, act='relu', name='block_1')
    #    [batch-size, npoints, 3, 64]
    #x = _block_conv2d(x, 64, reuse=reuse, is_training=is_training, act='relu', name='block_2')
    #    [batch-size, npoints, 3, 128]
    #x = _block_conv2d(x, 128, reuse=reuse, is_training=is_training, act='relu', name='block_3')
    ##    [batch-size, npoints, 3, 256]
    #x = _block_conv2d(x, 256, reuse=reuse, is_training=is_training, act='relu', name='block_4')
    ##    [batch-size, npoints, 3, 128]
    #x = _block_conv2d(x, 128, reuse=reuse, is_training=is_training, act='relu', name='block_5')
    #    [batch-size, npoints, 3, 64]
    #x = _block_conv2d(x, 64, reuse=reuse, is_training=is_training, act='relu', name='block_6')
    #    [batch-size, npoints, 3, 16]
    #x = _block_conv2d(x, 16, reuse=reuse, is_training=is_training, act='relu', name='block_7')
    #    [batch-size, npoints, 3, 6]
    #x = _block_conv2d(x, 6, reuse=reuse, is_training=is_training, act='relu', name='block_8')
    #    [batch-size, npoints, 3, 6]
    x = _block_conv2d(x, 8, kshape=9, reuse=reuse, is_training=is_training, act='relu', name='block_9', reshape=False)
    #    [batch-size, npoints, 18]
    #=>  [batch-size, 18, npoints]
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-2')
    x = layers.actives.squash(x, axis=1, epsilon=1e-9, reuse=reuse, safe=True, name='squash')
    x = layers.capsules.dense(x, nclass, 18, name='capsules', reuse=reuse, epsilon=1e-10, act='squash')
    #    [batch-size, 18, nclass]
    x = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-10)
    #    [batch-size, 18, nclass]
    return x

def get_tfrecord_size(filename):
    count = 0
    for _ in tf.python_io.tf_record_iterator(filename):
        count += 1
    return count

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
    if database is None or database == 'shapenet_part':
        from dataset.shapenet_part import parse
        train_filename = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized.tfrecord'
        valid_filename = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_valid.tfrecord'
        tests_filename = '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_test.tfrecord'
    else:
        from dataset.modelnet40 import parse
        train_filename = '/home/xiaox/studio/db/modelnet/train_normalize.tfrecord'
        valid_filename = None
        tests_filename = '/home/xiaox/studio/db/modelnet/test_normalize.tfrecord'

    engine.set_print(False)

    filename = tf.placeholder(tf.string, shape=[])
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse(num_points))
    dataset = dataset.shuffle(1000).batch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator = dataset.make_initializable_iterator()
    inputs, labels = iterator.get_next()
    #ops.core.summarize('inputs', inputs)
    def _build_net(inputs, reuse, is_training):
        return point_capsule_tio(inputs,
                                 is_training,
                                 nclass,
                                 reuse=reuse)

    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.allow_soft_placement = True
    with ops.core.device('/gpu:0'):
        trainp = _build_net(inputs, reuse=False, is_training=True)
        print(ops.core.shape(trainp))
        print(ops.core.shape(labels))
        train_loss_op = layers.losses.get('margin_loss', trainp, labels)
        ops.core.summarize('train_loss', train_loss_op)
        train_metric = layers.metrics.accuracy([trainp, labels])
        train_metric_op, train_metric_update_op, train_metric_initialize_op = train_metric
        ops.core.summarize('train_metric', train_metric_op)
        train_iters = int(get_tfrecord_size(train_filename) / batch_size)
        learning_rate = tf.train.exponential_decay(lr, global_step, train_iters, 0.9)
        update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
        with ops.core.control_dependencies(update_ops):
            train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(train_loss_op, global_step=global_step)

        validp = _build_net(inputs, reuse=True, is_training=False)
        valid_loss_op = layers.losses.get('margin_loss', validp, labels)
        ops.core.summarize('valid_loss', valid_loss_op)
        valid_metric = layers.metrics.accuracy([validp, labels])
        valid_metric_op, valid_metric_update_op, valid_metric_initialize_op = valid_metric
        ops.core.summarize('valid_metric', valid_metric_op)
        if valid_filename is not None:
            valid_iters = int(get_tfrecord_size(valid_filename) / batch_size)
        test_iters  = int(get_tfrecord_size(tests_filename) / batch_size)
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
            ops.core.run(sess, [train_metric_initialize_op, iterator.initializer], feed_dict={filename: train_filename})
            for iters in range(train_iters):
                _, loss, _, summary = ops.core.run(sess, [train_op, train_loss_op, train_metric_update_op, summarize])
                accuracy = ops.core.run(sess, train_metric_op)
                ops.core.add_summary(writer, summary, global_step=(epoch*train_iters)+iters)
                if iters % 10 == 0:
                    print('train for {}-th iteration: loss: {}, accuracy: {}'.format(iters, loss, accuracy))
            end = time.time()
            print('time cost:', end-start)
            # validation
            if valid_filename is not None:
                valid_loss = []
                valid_acc = []
                ops.core.run(sess, [valid_metric_initialize_op, iterator.initializer], feed_dict={filename: valid_filename})
                for iters in range(valid_iters):
                    loss, _ = ops.core.run(sess, [valid_loss_op, valid_metric_update_op])
                    accuracy = ops.core.run(sess, valid_metric_op)
                    valid_loss.append(loss)
                    valid_acc.append(accuracy)
                vloss = np.mean(valid_loss)
                vacc = np.mean(valid_acc)
                losses[epoch][0] = vloss
                losses[epoch][1] = vacc
                print('valid for {}-th epoch: loss:{}, accuracy: {}'.format(epoch, vloss, vacc))
            # test
            test_loss = []
            test_acc = []
            ops.core.run(sess, [valid_metric_initialize_op, iterator.initializer], feed_dict={filename: tests_filename})
            for iters in range(test_iters):
                loss, _ = ops.core.run(sess, [valid_loss_op, valid_metric_update_op])
                accuracy = ops.core.run(sess, valid_metric_op)
                test_loss.append(loss)
                test_acc.append(accuracy)
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
