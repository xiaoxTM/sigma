import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers, dbs, ops, engine, colors, helpers

import numpy as np

import tensorflow as tf

import logging

logging.basicConfig(level=logging.INFO)

def build_func(inputs, labels, initializer='glorot_normal'):
    # inputs shape :
    #    [batch-size, rows, cols, depth]
    # ops.core.summarize('inputs', inputs)
    x = layers.convs.conv2d(inputs, 256, 9,
                            padding='valid',
                            act='relu',
                            weight_initializer=initializer)
    # ops.core.summarize('conv2d-0', x)
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
                                         # stride=2,
                                         # activation for pre-predictions
                                         # that is, u^{hat}_{j|i}
                                         #act='leaky_relu',
                                         safe=True,
                                         weight_initializer=initializer,
                                         return_shape=True)
    # ops.core.summarize('conv2d-1', x)
    # x, outshape = layers.capsules.conv2d(x, 64, 32, 5, 1,
    #                                      stride=2,
    #                                      weight_initializer=initializer,
    #                                      return_shape=True)
    # ops.core.summarize('conv2d-2', x)

    #  [batch-size, nrows, ncols, 32, 8]
    #+>[batch_size, nrows * ncols * 32, 8]
    x = layers.base.reshape(x,
                            [outshape[0],
                             np.prod(outshape[1:-1]),
                             outshape[-1]])
    #  [batch_size, nrows * ncols * 32, 8]
    #=>[batch_size, 10, 16]
    x = layers.capsules.dense(x, 10, 16, 3,
                              weight_initializer=initializer)
    # ops.core.summarize('fully_connected-0', x)
    # norm the output to represent the existance probabilities
    # of each class
    classification = layers.capsules.norm(x)
    # ops.core.summarize('norm-0', classification)
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
    # ops.core.summarize('loss', loss, 'scalar')
    # ops.core.summarize('acc', metric[0], 'scalar')
    return [loss, metric]

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
gpu_config.gpu_options.visible_device_list = '0,1,2,3'
gpu_config.intra_op_parallelism_threads = 1

# sigma.engine.set_print(True, True)
nclass = 10
experiment, parser = sigma.build_experiment(build_func,
                                            engine.io.mnist('data',
                                                            False,
                                                            False,
                                                            nclass),
                                            'AdamOptimizer',
                                            # filename='mnist-networks.png',
                                            batch_size=100,
                                            # gpu_config=gpu_config,
                                            generator_config={'nclass':nclass})


if __name__=='__main__':
    args = parser.parse_args()
    args.checkpoint='cache'
    # args.log='cache'
    # args.auto_timestamp = True
    experiment(args)
