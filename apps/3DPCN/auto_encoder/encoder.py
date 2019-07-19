#! /usr/bin/env python3

import sys
import os.path

base = os.path.dirname(os.path.abspath(__file__))
# point to sigma parent directory
sys.path.append(os.path.abspath(os.path.join(base, '../../../../')))

import sigma
from sigma import layers

def encode(inputs, ncapsules):
    ''' inputs should have shape of:
        [batch-size, N, 3]
    '''
    # x: [batch-size, N, 3]
    #=>: [batch-size, N, 32]
    x = layers.convs.conv1d(inputs, 32, 1)
    x = layers.norms.batch_norm(x)
    x = layers.actives.relu(x)

    # x: [batch-size, N, 32]
    #=>: [batch-size, N, 128]
    x = layers.convs.conv1d(x, 128, 1)
    x = layers.norms.batch_norm(x)
    x = layers.actives.relu(x)

    # x: [batch-size, N, 128]
    #=>: [batch-size, N, 1024] * nprimary
    capsules = [layers.pools.max_pool1d(layers.norms.batch_norm(layers.conv1d(x, 1024, 1)))
                for _ in range(ncapsules)]

    # primaries: [batch-size, N, 1024] * nprimary
    #      x =>: [batch-size, N, 1024, nprimary]
    x = layers.merge.stack(capsules, axis=2)
    x = ops.core.squeeze(x)
    x = layers.actives.squash(x)

    return x

    #x = ops.capsules._agreement_routing(x, [-1, 1, 1024, 64, 1], 3, )


