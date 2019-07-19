#! /usr/bin/env python3

import sys
import os.path

base = os.path.dirname(os.path.abspath(__file__))
# point to sigma parent directory
sys.path.append(os.path.abspath(os.path.join(base, '../../../../')))

import sigma
from sigma import layers

def generate(x, bottleneck_size=2500):
    x = layers.convs.conv1d(x, bottleneck_size, 1)
    x = layers.norms.batch_norm(x)
    x = layers.actives.relu(x)

    x = layers.convs.conv1d(x, int(bottleneck_size / 2), 1)
    x = layers.norms.batch_norm(x)
    x = layers.actives.relu(x)

    x = layers.convs.conv1d(x, int(bottleneck_size / 4), 1)
    x = layers.norms.batch_norm(x)
    x = layers.actives.relu(x)

    x = layers.convs.conv1d(x, 3, 1)
    x = layers.actives.tanh(x)

    return x


def decode(inputs, ncapsules):
    capsules = []

    for i in range(ncapsules):
        grid = core.get_variable()
        o = generate(inputs)
        o = layers.merge.concat([o, grid], axis=1)
        capsules.append(o)
    return layers.merge.concat(capsules, axis=2)
