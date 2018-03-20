import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers
from sigma import dbs, ops, engine, colors

# import matplotlib.pyplot as plt

import tensorflow as tf


def build_func(inputs, labels,
               mode='depthwise'):
    x = layers.convs.conv2d(inputs, 256, 9, padding='valid', act='relu')
    x = layers.base.expand_dims(x, -2)
    # no routing between conv1 and primary caps
    # routing means:
    #     `reaching a agreement for all capsules in lower layer`
    # since the lower contains ONLY one capsule
    # we disable routing by setting iterations to 1
    x, outshape = layers.capsules.conv2d(x, 32, 8, 9, 1,
                                         stride=2,
                                         return_shape=True,
                                         mode=mode)
    x = layers.base.reshape(x, [outshape[0], -1, outshape[-1]])
    x = layers.capsules.dot(x, 10, 16, 3)
    # norm the output to represent the existance probabilities
    # of each class
    classification = layers.capsules.norm(x)
    class_loss = layers.losses.get('margin_loss', classification, labels)
    # reconstruction
    x = layers.base.maskout(x, axis=-2)
    x = layers.convs.dense(x, 512, act='relu')
    x = layers.convs.dense(x, 1024, act='relu')
    x = layers.convs.dense(x, 784, act='sigmoid')
    reconstruction = layers.base.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('reconstruction', reconstruction, max_outputs=10)
    recon_loss = layers.losses.mse([reconstruction, inputs])
    loss = layers.merge.add([class_loss, recon_loss], [1, 0.005])
    metric = layers.metrics.accuracy([classification, labels])
    return [loss, metric]


def train(xtrain, ytrain,
          xvalid, yvalid,
          checkpoints,
          nclass=10,
          epochs=30,
          batch_size=64,
          mode='depthwise',
          shuffle=True):
    input_shape = list(xtrain.shape)
    # sigma.engine.set_print(True)
    input_shape[0] = batch_size
    [xtensor, ytensor], [loss, metric] = sigma.build(input_shape,
                                                     build_func,
                                                     [batch_size, nclass],
                                                     mode=mode)
    tf.summary.scalar('loss', loss)
    # sigma.engine.export_graph('cache/network-architecture.png')

    #----- log optimizer gradients -----
    #optimizer = tf.train.AdamOptimizer(0.05)
    #grads_and_vars = optimizer.compute_gradients(loss)
    #for (grad, var) in grads_and_vars:
    #    if grad is not None:
    #        tf.summary.histogram(grad.name, grad)
    #optimizer = optimizer.apply_gradients(grads_and_vars)
    #----------

    optimizer = tf.train.AdamOptimizer(0.05)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '0, 1'
    config.intra_op_parallelism_threads = 1

    logs = 'cache/logs/capsulewise'
    if fastmode:
        logs = 'cache/logs/depthwise'

    sigma.train(xtrain,
                xtensor,
                optimizer,
                loss,
                metric,
                ytrain,
                ytensor,
                nclass=10,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                valids=[xvalid, yvalid],
                config=config,
                checkpoints=checkpoints,
                logs=logs,
                savemode='min')


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', type=str,
                    default='cache/checkpoints/capsule')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--shuffle', type=bool, default=True)
parser.add_argument('--mode', type=str, default='depthwise')

if __name__=='__main__':
    args = parser.parse_args()
    print('checkpoints: {}'.format(colors.red(args.checkpoints)))
    print('epochs: {}'.format(colors.red(args.epochs)))
    print('batch size: {}'.format(colors.red(args.batch_size)))
    print('shuffle: {}'.format(colors.red(args.shuffle)))
    print('mode: {}'.format(colors.red(args.mode)))
    (xtrain, ytrain), (xvalid, yvalid) = dbs.images.mnist.load(
        '/home/xiaox/studio/db/mnist', to_tensor=False)
    train(xtrain, ytrain,
          xvalid, yvalid,
          args.checkpoints,
          epochs=args.epochs,
          batch_size=args.batch_size,
          mode=args.mode,
          shuffle=args.shuffle)
