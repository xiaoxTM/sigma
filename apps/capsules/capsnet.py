import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers
from sigma import dbs, ops, engine

import tensorflow as tf


def build_func(x):
    x = layers.convs.conv2d(x, 256, 9, padding='valid', act='relu')
    x = layers.base.expand_dims(x, -2)
    # no routing between conv1 and primary caps
    # routing means:
    #     `reaching a agreement for all capsules in lower layer`
    # since the lower contains ONLY one capsule
    # we disable routing by setting iterations to 1
    x, outshape = layers.capsules.conv2d(x, 32, 8, 9, 1,
                                         stride=2,
                                         return_shape=True)
    x = layers.base.reshape(x, [outshape[0], -1, outshape[-1]])
    return layers.capsules.fully_connected(x, 10, 16, 3)


def train(xtrain, ytrain, checkpoints,
          nclass=10,
          epochs=3,
          batch_size=100,
          shuffle=True):
    input_shape = list(xtrain.shape)
#    sigma.engine.set_print(True)
    input_shape[0] = batch_size
    ytensor = sigma.placeholder(dtype=ops.core.int32, shape=[batch_size, nclass])
    xtensor, loss = sigma.build(input_shape,
                                build_func,
                                'margin_loss',
                                labels=ytensor,
                                onehot=True)
    tf.summary.scalar('loss', loss)
#    graph = tf.get_default_graph()
#    collections = ['summaries', 'variables', 'trainable_variables']
#    print(collections)
#    for collection in collections:
#        print(collection)
#        gparams = tf.get_collection(collection)
#        if len(gparams) == 0:
#            print('ERROR: {} has no parameters to optimize. \
#                  make sure you set `scope` to layers with trainable parameters'.format(collection))
#        print('parameters for {} training'.format(collection))
#        for gp in gparams:
#            print('    ', gp.name, gp.get_shape().as_list())

#    sigma.engine.export_graph('cache/network-architecture.png')

    #----- log optimizer gradients -----
    optimizer = tf.train.AdamOptimizer(0.05)
    grads_and_vars = optimizer.compute_gradients(loss)
    for (grad, var) in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(grad.name, grad)
    optimizer = optimizer.apply_gradients(grads_and_vars)
    #----------
    #optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '0'
    config.intra_op_parallelism_threads = 1

    sigma.run(xtrain, xtensor,
              optimizer,
              loss,
              ytrain, ytensor,
              nclass=10,
              epochs=epochs,
              batch_size=batch_size,
              shuffle=shuffle,
              checkpoints=checkpoints,
              config=config,
              logs='cache/logs',
              save='min')


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints', type=str, default='cache/checkpoints/capsule')

if __name__=='__main__':
    args = parser.parse_args()
    (xtrain, ytrain), (xvalid, yvalid) = dbs.images.mnist.load('/home/xiaox/studio/db/mnist',
                                                               to_tensor=False)
    train(xtrain, ytrain, args.checkpoints)
