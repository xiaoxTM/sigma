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
    x = layers.convs.conv2d(inputs, 256, 3, padding='valid', act='relu')
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
    x = layers.capsules.conv2d(x, 16, 8, 3, 1,
                               stride=1,
                               # activation for pre-predictions
                               # that is, u^{hat}_{j|i}
                               #act='leaky_relu',
                               safe=True)
    ops.core.summarize('conv2d-1', x)
    x = layers.capsules.conv2d(x, 24, 12, 3, 3,
                               stride=1,
                               safe=True)
    x, outshape = layers.capsules.conv2d(x, 32, 16, 3, 3,
                                         stride=2,
                                         safe=True,
                                         return_shape=True)
    x = layers.base.reshape(x,
                            [outshape[0],
                             np.prod(outshape[1:-1]),
                             outshape[-1]])
    x = layers.capsules.dense(x, 20, 16, 3, safe=True)
    ops.core.summarize('fully_connected-0', x)
    # norm the output to represent the existance probabilities
    # of each class
    classification = layers.capsules.norm(x, safe=True)
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
    ops.core.summarize('loss', loss, 'scalar')
    ops.core.summarize('acc', metric[0], 'scalar')
    return [loss, metric]

gpu_config = tf.ConfigProto()
gpu_config.gpu_options.allow_growth = True
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.8
gpu_config.gpu_options.visible_device_list = '0,1'
gpu_config.intra_op_parallelism_threads = 1


cifardb = '/home/xiaox/studio/db/cifar/cifar-100-python'
nclass = 20
generator_config={'nclass':nclass}
sigma.engine.set_print(True, True)
experiment, parser = sigma.build_experiment(build_func,
                                            engine.io.cifar(cifardb,
                                                            False,
                                                            nclass=nclass),
                                            'AdamOptimizer',
                                            batch_size=40,
                                            filename='cifar-100-capsules.png',
                                            generator_config=generator_config,
                                            gpu_config=gpu_config)


if __name__=='__main__':
    # add some option here if necessary
    # > parser.add_argument('--print', type=bool, default=True):
    exp = '/home/xiaox/studio/exp/sigma/capsules/dynamic-routing/cifar'
    args = parser.parse_args()
    args.checkpoint = os.path.join(exp, 'cache')
    args.log = os.pathjoin(exp, 'cache')
    experiment(args)
