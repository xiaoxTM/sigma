import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, dbs, ops, engine, colors, helpers

import numpy as np

import tensorflow as tf


def build_func(inputs, labels):
    # inputs shape :
    #    [batch-size, rows, cols, depth]
    ops.core.summarize('inputs', inputs)
    x = layers.convs.conv2d(inputs, 256, 9, padding='valid', act='relu')
    ops.core.summarize('conv2d-0', x)
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
    x, outshape = layers.capsules.conv2d(x, 32, 8, 9, 1,
                                         stride=2,
                                         # activation for pre-predictions
                                         # that is, u^{hat}_{j|i}
                                         act='leaky_relu',
                                         return_shape=True)
    ops.core.summarize('conv2d-1', x)
    #x, outshape = layers.capsules.conv2d(x, 64, 32, 5, 1,
    #                                     stride=1,
    #                                     return_shape=True,
    #                                     mode=mode)
    x = layers.base.reshape(x, [outshape[0], np.prod(outshape[1:-1]), outshape[-1]])
    x = layers.capsules.dense(x, 10, 16, 3)
    ops.core.summarize('dense', x)
    # norm the output to represent the existance probabilities
    # of each class
    classification = layers.capsules.norm(x)
    ops.core.summarize('norm', classification)
    class_loss = layers.losses.get('margin_loss', classification, labels)
    loss = class_loss
    #tf.summary.scalar('classification-loss', class_loss)
    ## reconstruction
    #x = layers.base.maskout(x, axis=-2)
    #x = layers.convs.dense(x, 512, act='relu')
    #x = layers.convs.dense(x, 1024, act='relu')
    #x = layers.convs.dense(x, 784, act='sigmoid')
    #reconstruction = layers.base.reshape(x, [-1, 28, 28, 1])
    #tf.summary.image('reconstruction', reconstruction, max_outputs=10)
    #recon_loss = layers.losses.mse([reconstruction, inputs])
    #tf.summary.scalar('reconstruction-loss', recon_loss)
    #loss = layers.merge.add([class_loss, recon_loss], [1, 0.005])
    metric = layers.metrics.accuracy([classification, labels])
    return [loss, metric]


# def train(xtrain, ytrain,
#           xvalid, yvalid,
#           checkpoints,
#           nclass=10,
#           epochs=30,
#           batch_size=64,
#           expid=None,
#           shuffle=True):
#     input_shape = list(xtrain.shape)
#     # sigma.engine.set_print(True)
#     input_shape[0] = batch_size
#     [xtensor, ytensor], [loss, metric] = sigma.build(input_shape,
#                                                      build_func,
#                                                      [batch_size, nclass])
#     tf.summary.scalar('loss', loss)
#     tf.summary.scalar('metric', metric[0])
#     # sigma.engine.export_graph('cache/network-architecture.png')
#
#     #----- log optimizer gradients -----
#     #optimizer = tf.train.AdamOptimizer(0.05)
#     #grads_and_vars = optimizer.compute_gradients(loss)
#     #for (grad, var) in grads_and_vars:
#     #    if grad is not None:
#     #        tf.summary.histogram(grad.name, grad)
#     #optimizer = optimizer.apply_gradients(grads_and_vars)
#     #----------
#
#     if expid is None:
#         expid = helpers.timestamp()
#     else:
#         expid = '{}-{}'.format(helpers.timestamp(), expid)
#
#     optimizer = tf.train.AdamOptimizer()
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     config.gpu_options.per_process_gpu_memory_fraction = 0.8
#     config.gpu_options.visible_device_list = '0, 1'
#     config.intra_op_parallelism_threads = 1
#
#     logs = 'cache/logs/capsule/{}'.format(expid)
#     checkpoints = '{}/{}/main.ckpt'.format(checkpoints, expid)
#     sigma.train(xtrain,
#                 xtensor,
#                 optimizer,
#                 loss,
#                 metric,
#                 ytrain,
#                 ytensor,
#                 nclass=10,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 shuffle=shuffle,
#                 valids=[xvalid, yvalid],
#                 config=config,
#                 checkpoints=checkpoints,
#                 logs=logs,
#                 savemode='min')
#
#
# parser = argparse.ArgumentParser()
# parser.add_argument('--checkpoints', type=str,
#                     default='cache/checkpoints/capsule')
# parser.add_argument('--epochs', type=int, default=100)
# parser.add_argument('--batch_size', type=int, default=100)
# parser.add_argument('--shuffle', type=bool, default=True)
# parser.add_argument('--expid', type=str, default=None)

experiment, parser = sigma.build_experiment(
    build_func,
    dbs.images.mnist.load,
    'AdamOptimizer',
    nclass=10,
    reader_config={'dirs':'/home/xiaox/studio/db/mnist',
                   'to_tensor':False,
                   'onehot':False})


if __name__=='__main__':
    # args = parser.parse_args()
    experiment(parser)
