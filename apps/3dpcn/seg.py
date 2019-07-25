#! /usr/bin/env python3

import sys
import os
import time
import os.path

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../../'))
sys.path.append(root)

from auto_encoder import point_capsule_seg, point_capsule_net
from sigma import layers, engine, ops, status
from dataset import shapenet_part
import argparse

parser = argparse.ArgumentParser(description='3D point capsule network implementation with TensorFlow')
parser.add_argument('--phase', default='train', type=str, help='"train" or "eval" mode switch')
parser.add_argument('--batch-size', default=8, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--channels', default=1, type=int)
parser.add_argument('--num-points', default=2048, type=int)
parser.add_argument('--num-latent', default=64, type=int)
parser.add_argument('--vec-latent', default=64, type=int)
parser.add_argument('--primary-size', default=16, type=int)
parser.add_argument('--nclass', default=8, type=int)
parser.add_argument('--checkpoint', default='/home/xiaox/studio/exp/3dpcn/checkpoint', type=str)
parser.add_argument('--logdir', default=None, type=str)
parser.add_argument('--dataset',
        default='/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0', type=str)
parser.add_argument('--normalize', default=True, type=bool)
parser.add_argument('--shuffle', default=True, type=bool)


def train_net(args):
    engine.set_print(True, )
    inputs = layers.base.input_spec([None, args.num_points, 3])
    def _build_net(reuse):
        return point_capsule_seg(inputs,
                                 args.batch_size,
                                 args.primary_size,
                                 args.num_latent,
                                 args.vec_latent,
                                 args.channels,
                                 reuse=reuse)

    global_step = ops.core.get_variable('global-step',
                                        initializer=0,
                                        trainable=False)
    with ops.core.device('/gpu:0'):
        x = _build_net(reuse=False)
        _, reconstruction = x
        loss_op = layers.losses.chamfer_loss([inputs, reconstruction])
        train_op = ops.optimizers.get('AdamOptimizer').minimize(loss_op, global_step=global_step)
        update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
        train_op = ops.core.group(train_op, update_ops)

        status.set_phase('valid')
        valid_x = _build_net(reuse=True)
        _, valid_recons = valid_x
        valid_loss_op = layers.losses.chamfer_loss([inputs, valid_recons])

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    sess, saver, _, _ = engine.session(checkpoint=args.checkpoint,
                                       log=args.logdir)

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
    layers.core.export_graph('auto_encoder.png')
    with sess:
        for epoch in range(args.epochs):
            start = time.time()
            status.set_phase('train')
            for iters in range(train_iters):
                trainx, train_pid, train_oid = next(train_db)
                _, loss = ops.core.run(sess, [train_op, loss_op], {inputs:trainx})
                if iters % 10 == 0:
                    print('train for {}-th iteration: loss: {}'.format(iters, loss))
            end = time.time()
            print('time cost:', end-start)
            # validation
            status.set_phase('valid')
            valid_loss = []
            for iters in range(valid_iters):
                validx, valid_pid, valid_oid = next(valid_db)
                loss = ops.core.run(sess, valid_loss_op, {inputs:validx})
                valid_loss.append(loss)
            vloss = np.mean(valid_loss)
            print('valid for {}-th epoch: loss:{}'.format(epoch, vloss))
        if epoch % 10 == 0:
            helpers.save(sess, args.checkpoint, saver, False)

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
