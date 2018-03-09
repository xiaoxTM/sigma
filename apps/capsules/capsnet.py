import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers
from sigma import dbs, ops, engine

import tensorflow as tf

def build(input_shape, loss, reuse=False, scope=None, **kwargs):
    with sigma.defaults(reuse=reuse, scope=scope):
        inputs = layers.base.input_spec(input_shape, dtype='float32', name='input-data')
        conv1 = layers.convs.conv2d(inputs, 256, 9, padding='valid', act='relu')
        conv1_expanded = layers.base.expand_dims(conv1, -2)
        primary_caps, outshape = layers.capsules.conv2d(conv1_expanded, 32, 8, 9, 3,
                                                        stride=2,
                                                        return_shape=True)
        # outshape : [batch-size, rows, cols, outcaps, outcapdim]
        #         => [batch-size, -1, outcapdim]
        primary_reshaped = layers.base.reshape(primary_caps, [outshape[0], -1, outshape[-1]])
        digit_caps = layers.capsules.fully_connected(primary_reshaped, 10, 16, 3)
        return inputs, layers.losses.get(loss, digit_caps, **kwargs)


def train(xtrain, ytrain, checkpoints,
          nclass=10,
          epochs=1,
          batch_size=100,
          shuffle=True):
    input_shape = list(xtrain.shape)
#    sigma.engine.set_print(True)
    input_shape[0] = batch_size
    ytensor = sigma.placeholder(dtype=ops.core.int32, shape=[batch_size, nclass])
    xtensor, loss = build(input_shape, 'margin_loss', labels=ytensor, onehot=True)
#    loss = layers.losses.margin_loss(preds, ytensor, onehot=True)
#    sigma.engine.export_graph('cache/network-architecture.png')
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.visible_device_list = '0'
    config.intra_op_parallelism_threads = 1

    tf.summary.scalar('loss', loss)

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
