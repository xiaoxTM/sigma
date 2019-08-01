import sys
import os.path

# add sigma to sys path
curd = os.path.dirname(os.path.abspath(__file__))
root = os.path.realpath(os.path.join(curd, '../../../'))
sys.path.append(root)

from sigma import layers, ops

from .encoder import encode
from .decoder import decode

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
