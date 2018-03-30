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
                                        #  act='leaky_relu',
                                         return_shape=True)
    ops.core.summarize('conv2d-1', x)
    #x, outshape = layers.capsules.conv2d(x, 64, 32, 5, 1,
    #                                     stride=1,
    #                                     return_shape=True,
    #                                     mode=mode)
    x = layers.base.reshape(x, [outshape[0], np.prod(outshape[1:-1]), outshape[-1]])
    x = layers.capsules.dense(x, 20, 16, 3)
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


reader_config={'dirs':'/home/xiaox/studio/db/cifar/cifar-100-python',
               'to_tensor':False,
               'onehot':False}
experiment, parser = sigma.build_experiment(build_func,
                                            dbs.images.cifar.load,
                                            'AdamOptimizer',
                                            nclass=20,
                                            reader_config=reader_config)


if __name__=='__main__':
    # add some option here if necessary
    # > parser.add_argument('--print', type=bool, default=True)
    experiment(parser)
