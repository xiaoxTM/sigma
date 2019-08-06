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

from auto_encoder import point_capsule_seg, point_capsule_net
from sigma import layers, engine, ops, status, helpers
from dataset import shapenet_part
import argparse

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=6, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--num-latent', default=64, type=int)
parser.add_argument('--vec-latent', default=64, type=int)
parser.add_argument('--primary-size', default=16, type=int)
parser.add_argument('--nclass', default=8, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/cache/checkpoint/model.ckpt', type=str)
parser.add_argument('--logdir', default='/home/xiaox/studio/exp/3dpcn/cache/log', type=str)
#parser.add_argument('--address', default=None)
parser.add_argument('--address', default='172.31.234.152:2666')
parser.add_argument('--dataset',
        default='/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--debug', default=False, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)

#def build_net(args, reuse=False):
#    x = point_capsule_net(inputs,
#                          args.batch_size,
#                          args.primary_size,
#                          args.num_latent,
#                          args.vec_latent,
#                          args.channels,
#                          reuse=reuse)
#    return inputs, x

def train_net(args):
    engine.set_print(None, )
    inputs = layers.base.input_spec([None, args.num_points, 3])
    ops.core.summarize('inputs', inputs)
    def _build_net(reuse, is_training):
        return point_capsule_net(inputs,
                                 is_training,
                                 args.batch_size,
                                 args.primary_size,
                                 args.num_latent,
                                 args.vec_latent,
                                 args.channels,
                                 reuse=reuse)

    train_db, train_iters = shapenet_part.dataset(root=args.dataset,
                                     splits='shuffled_train_file_list.json',
                                     batchsize=args.batch_size,
                                     num_points=args.num_points,
                                     shuffle=args.shuffle,
                                     normalize=args.normalize)
    valid_db, valid_iters = shapenet_part.dataset(root=args.dataset,
                                     splits='shuffled_val_file_list.json',
                                     batchsize=args.batch_size,
                                     num_points=args.num_points,
                                     shuffle=args.shuffle,
                                     normalize=args.normalize)
    test_db, test_iters = shapenet_part.dataset(root=args.dataset,
                                     splits='shuffled_test_file_list.json',
                                     batchsize=args.batch_size,
                                     num_points=args.num_points,
                                     shuffle=args.shuffle,
                                     normalize=args.normalize)

    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)
    with ops.core.device('/gpu:0'):
        train_net = _build_net(reuse=False, is_training=True)
        _, train_reconstruction = train_net
        loss_op = layers.losses.chamfer_loss([inputs, train_reconstruction], dtype=ops.core.float32, alpha=1, belta=1)
        learning_rate = tf.train.exponential_decay(0.001, global_step, train_iters, 0.9)
        update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
        with ops.core.control_dependencies(update_ops):
            train_op = ops.optimizers.get('AdamOptimizer', learning_rate=learning_rate).minimize(loss_op, global_step=global_step)

        valid_net = _build_net(reuse=True, is_training=False)
        _, valid_reconstruction = valid_net
        valid_loss_op = layers.losses.chamfer_loss([inputs, valid_reconstruction], dtype=ops.core.float32, alpha=1,
                belta=1)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.allow_soft_placement = True
    sess, saver, summarize, writer = engine.session(checkpoint=args.checkpoint,
                                                    config=config,
                                                    debug=args.debug,
                                                    address=args.address,
                                                    log=args.logdir)

    base = '/home/xiaox/studio/exp/3dpcn'
    #layers.core.export_graph('auto_encoder.png')
    with sess:
        losses =  np.zeros((args.epochs, 2))
        for epoch in range(args.epochs):
            start = time.time()
            #status.set_phase('train')
            for iters in range(train_iters):
                trainx, train_pid, train_oid = next(train_db)
                _, loss, summary = ops.core.run(sess, [train_op, loss_op, summarize], {inputs:trainx})
                ops.core.add_summary(writer, summary, global_step=(epoch*train_iters)+iters)
                if iters % 10 == 0:
                    print('train for {}-th iteration: loss: {}'.format(iters, loss))
            end = time.time()
            print('time cost:', end-start)
            # validation
            #status.set_phase('valid')
            #valid_loss = []
            #for iters in range(valid_iters):
            #    validx, valid_pid, valid_oid = next(valid_db)
            #    loss = ops.core.run(sess, valid_loss_op, {inputs:validx})
            #    valid_loss.append(loss)
            #vloss = np.mean(valid_loss)
            #losses[epoch][0] = vloss
            #print('valid for {}-th epoch: loss:{}'.format(epoch, vloss))
            #test_loss = []
            #for iters in range(test_iters):
            #    testx, test_pid, test_oid = next(test_db)
            #    loss, recons = ops.core.run(sess, [valid_loss_op, valid_recons], {inputs:testx})
            #    test_loss.append(loss)
            #    if epoch % 20 == 0:
            #        for idx, (ipt, tst) in enumerate(zip(testx, recons)):
            #            os.makedirs('{}/{}/{}'.format(base, epoch, iters), exist_ok=True)
            #            np.savetxt('{}/{}/{}/{}-input.txt'.format(base, epoch, iters, idx), ipt)
            #            np.savetxt('{}/{}/{}/{}-recst.txt'.format(base, epoch, iters, idx), tst)
            #tloss = np.mean(test_loss)
            #losses[epoch][1] = tloss
            #print('test for {}-th epoch: loss:{}'.format(epoch, tloss))
            if epoch % 10 == 0:
                helpers.save(sess, args.checkpoint, saver, True, global_step=epoch)
        np.savetxt('losses.log', losses)
        ops.core.close_summary_writer(writer)

def eval_net(args):
    inputs, x = build_net(args)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.phase == 'train':
        train_net(args)
    elif args.phase == 'eval':
        eval_net(args)
    else:
        raise ValueError('mode must be either `train` or `eval`. given `{}`'.format(args.mode))
