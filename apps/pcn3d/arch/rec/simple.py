import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import layers
from sigma.ops import core
from sigma.apps.pcn3d import dataset
from . import ops

def build_net(inputs, labels, loss='margin_loss', batch_size=32, nclass=16, reuse=False, is_training=True, **kwargs):
    with core.device('/gpu:0'):
        if is_training:
            inputs = dataset.rotate_point_cloud(inputs, batch_size=batch_size)
            inputs = dataset.jitter_point_cloud(inputs, batch_size=batch_size)
        #inputs: [batch-size, 2048, 3]
        #=>      [batch-size, 3, 2048]
        x = layers.base.transpose(inputs, (0, 2, 1))
        #        [batch-size, 3, 2048]
        #=>      [batch-size, 16, 512]
        core.summarize('inputs', x, reuse=reuse)
        x = ops.permutation_transform(x, 512, 16, mode='max', reuse=reuse, name='permutation_transform', act='squash')
        core.summarize('permutation_transform', x, reuse=reuse)
        #x = layers.capsules.dense(x, 256, 12, reuse=reuse, epsilon=1e-9, name='dense-1', act='relu')
        #if not reuse:
        #    ops.core.summarize('dense-1', x)
        #x = layers.capsules.dense(x, 128, 24, reuse=reuse, epsilon=1e-9, name='dense-2', act='relu')
        #if not reuse:
        #    ops.core.summarize('dense-2', x)
        #x = layers.capsules.dense(x,  64, 48, reuse=reuse, epsilon=1e-9, name='dense-3', act='relu')
        #if not reuse:
        #    ops.core.summarize('dense-3', x)
        #x = layers.capsules.dense(x,  32, 96, reuse=reuse, epsilon=1e-9, name='dense-4', act='relu')
        #if not reuse:
        #    ops.core.summarize('dense-4', x)
        x = layers.capsules.dense(x,  nclass, 24, reuse=reuse, epsilon=1e-9, name='dense-5', act='squash')
        core.summarize('dense-5', x, reuse=reuse)
        x = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-9, name='norm', reuse=reuse)
        core.summarize('norm', x, reuse=reuse)
        loss_op = layers.losses.get(loss, x, labels)
        ops.core.summarize('train-loss', loss_op, 'scalar')
        metric_op = layers.metrics.accuracy([x, labels])
    return loss_op, metric_op
