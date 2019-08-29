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

from sigma import layers, engine, ops, status, helpers
import argparse
import dataset
from arch import rec

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/cache/checkpoint/model.ckpt', type=str)
parser.add_argument('--log', default='/home/xiaox/studio/exp/3dpcn/cache/log', type=str)
parser.add_argument('--address', default='172.31.234.152:2666')
parser.add_argument('--database', default='shapenet_part', type=str)
parser.add_argument('--gpu', default='1', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--net-arch', default='simple', type=str)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--views', default=3, type=int)
parser.add_argument('--shuffle', default=True, type=bool)
parser.add_argument('--learning-rate', default=0.05, type=float)


@helpers.stampit({'checkpoint':-2, 'log':-1}, verbose=True)
def train_net(batch_size=8,
              epochs=1000,
              num_points=2048,
              views=3,
              lr=0.02,
              debug=False,
              address=None,
              checkpoint=None,
              log=None,
              net_arch='simple',
              database='shapenet_part',
              gpu='0'):
    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)
    nclass = 40
    if database == 'shapenet_part':
        nclass = 16

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    num_gpus = len(gpu.split(','))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.allow_soft_placement = True
    with ops.core.device('/gpu:0'):
        train_filename, valid_filename, tests_filename, \
        train_iters, valid_iters, tests_iters, \
        filename, iterator = dataset.prepare_dataset(num_points, batch_size, epochs, database)
        inputs, labels = iterator.get_next()
    train_loss_op, train_metric = rec.build_net(net_arch, inputs, labels, nclass=nclass, is_training=True, reuse=False, views=views, num_gpus=num_gpus)
    valid_loss_op, valid_metric = rec.build_net(net_arch, inputs, labels, nclass=nclass, reuse=True, is_training=False, views=views, num_gpus=num_gpus)
    learning_rate = tf.train.exponential_decay(lr, global_step, train_iters, 0.9)
    update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
    with ops.core.control_dependencies(update_ops):
        train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(train_loss_op, global_step=global_step)
    train_metric_op, train_metric_update_op, train_metric_initialize_op = train_metric
    valid_metric_op, valid_metric_update_op, valid_metric_initialize_op = valid_metric
    sess, saver, summarize, writer = engine.session(checkpoint=checkpoint,
                                                    config=config,
                                                    debug=debug,
                                                    address=address,
                                                    log=log)

    with sess:
        losses =  np.zeros((epochs, 4))
        print('train iteration:', train_iters)
        if valid_iters is not None:
            print('valid iteration:', valid_iters)
        print('tests iteration:', tests_iters)
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
            if valid_iters is not None:
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
            for iters in range(tests_iters):
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
        np.savetxt('losses/{}_{}.loss'.format(net_arch, database), losses)
        ops.core.close_summary_writer(writer)


if __name__ == '__main__':
    args = parser.parse_args()
    helpers.set_term_title(args.net_arch+"_"+args.database)
    if args.phase == 'train':
        train_net(args.batch_size,
                  args.epochs,
                  args.num_points,
                  args.views,
                  args.learning_rate,
                  args.debug,
                  args.address,
                  args.checkpoint,
                  args.log,
                  args.net_arch,
                  args.database,
                  args.gpu)
    else:
        raise ValueError('mode must be `train`. given `{}`'.format(args.phase))
