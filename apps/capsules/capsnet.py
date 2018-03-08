import argparse
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers
from sigma import dbs, ops

import tensorflow as tf

def build(input_shape, reuse=False, scope=None):
    with sigma.defaults(reuse=reuse, scope=scope):
        inputs = layers.base.input_spec(input_shape, dtype='float32', name='input-data')
        conv1 = layers.convs.conv2d(inputs, 256, 9, padding='valid', act='relu')
        conv1_expanded = ops.core.expand_dims(conv1, -2)
        primary_caps, outshape = layers.capsules.conv2d(conv1_expanded, 32, 8, 9, 3,
                                                        stride=2,
                                                        return_shape=True)
        # outshape : [batch-size, rows, cols, outcaps, outcapdim]
        #         => [batch-size, -1, outcapdim]
        primary_reshaped = layers.base.reshape(primary_caps, [outshape[0], -1, outshape[-1]])
        digit_caps = layers.capsules.fully_connected(primary_reshaped, 10, 16, 3)
        return inputs, digit_caps


def train(xtrain, ytrain, checkpoints,
          nclass=10,
          epochs=2,
          batch_size=100,
          shuffle=True):
    input_shape = list(xtrain.shape)
    print('train dataset shape:', input_shape)
    input_shape[0] = batch_size
    xtensor, preds = build(input_shape)
    ytensor = sigma.placeholder(dtype=ops.core.int32, shape=[batch_size, nclass])
    loss = layers.losses.margin_loss(preds, ytensor, onehot=True)
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
