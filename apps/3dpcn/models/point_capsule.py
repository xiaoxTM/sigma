import sys
import os.path

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../'))
sys.path.append(root)

from sigma import layers, ops

from .block import _block, _block_conv2d
from .encoder import encode
from .decoder import decode

def point_capsule_tio(inputs,
                      is_training,
                      nclass=40,
                      reuse=False):
    #    [batch-size, npoints, 3]
    npoints = ops.core.shape(inputs)[1]
    #    [batch-size, npoints, 3]
    #=>  [batch-size, 3, npoints]; `npoints` capsules, each of which has 3 dims
    x = layers.base.transpose(inputs, (0, 2, 1), reuse=reuse, name='transpose-0')
    x = layers.capsules.order_invariance_transform(x, npoints, 9, act='squash', name='tio', reuse=reuse)
    #    [batch-size, 18, npoints]
    #=>  [batch-size, npoints, 18]
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-1')
    #    [batch-size, npoints, 18, 1]
    x = layers.base.reshape(x, [-1, npoints, 9, 1], name='reshape', reuse=reuse)
    #    [batch-size, npoints, 3, 16]
    #x = _block_conv2d(x, 16, reuse=reuse, is_training=is_training, act='relu', name='block_1')
    #    [batch-size, npoints, 3, 64]
    #x = _block_conv2d(x, 64, reuse=reuse, is_training=is_training, act='relu', name='block_2')
    #    [batch-size, npoints, 3, 128]
    #x = _block_conv2d(x, 128, reuse=reuse, is_training=is_training, act='relu', name='block_3')
    ##    [batch-size, npoints, 3, 256]
    #x = _block_conv2d(x, 256, reuse=reuse, is_training=is_training, act='relu', name='block_4')
    ##    [batch-size, npoints, 3, 128]
    #x = _block_conv2d(x, 128, reuse=reuse, is_training=is_training, act='relu', name='block_5')
    #    [batch-size, npoints, 3, 64]
    #x = _block_conv2d(x, 64, reuse=reuse, is_training=is_training, act='relu', name='block_6')
    #    [batch-size, npoints, 3, 16]
    #x = _block_conv2d(x, 16, reuse=reuse, is_training=is_training, act='relu', name='block_7')
    #    [batch-size, npoints, 3, 6]
    #x = _block_conv2d(x, 6, reuse=reuse, is_training=is_training, act='relu', name='block_8')
    #    [batch-size, npoints, 3, 6]
    x = _block_conv2d(x, 8, kshape=9, reuse=reuse, is_training=is_training, act='relu', name='block_9', reshape=False)
    #    [batch-size, npoints, 18]
    #=>  [batch-size, 18, npoints]
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse, name='transpose-2')
    x = layers.actives.squash(x, axis=1, epsilon=1e-9, reuse=reuse, safe=True, name='squash')
    x = layers.capsules.dense(x, nclass, 18, name='capsules', reuse=reuse, epsilon=1e-10, act='squash')
    #    [batch-size, 18, nclass]
    x = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-10)
    #    [batch-size, 18, nclass]
    return x


def point_capsule_net(inputs,
                      is_training,
                      batchsize,
                      primary_size=16,
                      num_latent=64,
                      vec_latent=64,
                      channels=1,
                      reuse=False):
    # inputs should have the shape of
    #  [batch-size, N, 3]
    num_points = ops.core.shape(inputs)[1]

    # inputs: [batch-size, N, 3]
    #=>    x: [batch-size, 1024, primary-size]
    x = encode('pcn_encode', inputs, primary_size, reuse=reuse, is_training=is_training)

    #      x: [batch-size, 1024, primary-size]
    #=>    x: [batch-size, 64, 64]
    latent_capsules = layers.capsules.dense(x, num_latent, vec_latent, name='pcn_encode_dense', reuse=reuse, epsilon=1e-10)
    ops.core.summarize('pcn_encode_dense', latent_capsules)

    #      x: [batch-size, 64, 64]
    #=>    x: [batch-size, num_points, 3]
    x = decode('pcn_decode', latent_capsules, batchsize, num_points, channels=channels, reuse=reuse, is_training=is_training)
    return latent_capsules, x

def point_capsule_rec(inputs,
                      is_training,
                      primary_size=16,
                      vec_latent=64,
                      nclass=8,
                      channels=1,
                      reuse=False):

    # inputs should have the shape of
    #  [batch-size, N, 3]
    num_points = ops.core.shape(inputs)[1]

    # inputs: [batch-size, N, 3]
    #=>    x: [batch-size, 1024, primary-size]
    x = encode('pcn_encode', inputs, primary_size, reuse=reuse, is_training=is_training)

    #      x: [batch-size, 1024, primary-size]
    #=>    x: [batch-size, vec_latent, nclass]
    latent_capsules = layers.capsules.dense(x, nclass, vec_latent, name='pcn_encode_dense', reuse=reuse, epsilon=1e-10)
    ops.core.summarize('pcn_encode_dense', latent_capsules)

    #      x: [batch-size, vec_latent, nclass]
    #=>    x: [batch-size, nclass]
    prediction = layers.capsules.norm(latent_capsules, axis=1, safe=True, epsilon=1e-9)
    return prediction


def point_capsule_seg(inputs,
                      is_training,
                      primary_size=16,
                      num_latent=64,
                      vec_latent=64,
                      nclass=8,
                      channels=1,
                      reuse=False):
    # inputs should have the shape of
    #  [batch-size, N, 3]

    # inputs: [batch-size, N, 3]
    #=>    x: [batch-size, 1024, primary-size]
    x = encode('pcs_encode', inputs, primary_size, reuse=reuse, is_training=is_training)

    #      x: [batch-size, 1024, primary-size]
    #=>    x: [batch-size, 64, 64]
    latent_capsules = layers.capsules.dense(x, num_latent, vec_latent, name='pcs_encode_dense', reuse=reuse)
    ops.core.summarize('pcn_encode_dense', x)

    #      x: [batch-size, 64, 64]
    #=>    x: [batch-size, 64, nclass]
    x = layers.convs.conv1d(latent_capsules, nclass, name='pcs_encode_conv1d', reuse=reuse)
    ops.core.summarize('pcn_encode_conv1d', x)

    #      x: [batch-size, 64, nclass]
    #=>    x: [batch-size, nclass, 64]
    x = layers.base.transpose(x, (0, 2, 1), reuse=reuse)

    x = layers.actives.softmax(x, name='pcs_encode_softmax', reuse=reuse)
    ops.core.summarize('pcs_encode_softmax', x)

    x = ops.core.log(x)
    ops.core.summarize('pcs_encode_log', x)
    #     x: [batch-size, nclass, 64]
    #=>   x: [batch-size, ]
    x = layers.base.reshape(x, (-1, num_latent, nclass), name='pcs_encode_reshape', reuse=reuse)
    return x
