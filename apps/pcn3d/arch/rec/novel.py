import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import layers
from sigma.ops import core
from . import ops

def build_net(inputs, nclass=16, reuse=False):
    #inputs: [batch-size, 2048, 3]
    #=>      [batch-size, 3, 2048]
    x = layers.base.transpose(inputs, (0, 2, 1))
    #        [batch-size, 3, 2048]
    #=>      [batch-size, 16, 512]
    core.summarize('inputs', x, reuse=reuse)
    x = ops.permutation_transform(x, 512, 16, mode='max', reuse=reuse, name='permutation_transform', act='squash')
    core.summarize('permutation_transform', x, reuse=reuse)
    x = ops.project_transform(x, 32, reuse=reuse, name='projection', act='squash')
    x = layers.capsules.dense(x,  nclass, 24, reuse=reuse, epsilon=1e-9, name='dense-5', act='squash')
    core.summarize('dense-5', x, reuse=reuse)
    x = layers.capsules.norm(x, safe=True, axis=1, epsilon=1e-9, name='norm', reuse=reuse)
    core.summarize('norm', x, reuse=reuse)
    return x
