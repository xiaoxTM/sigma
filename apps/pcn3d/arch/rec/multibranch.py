import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import layers
from sigma.ops import core
from . import ops

def build_net(inputs, views=2, nclass=40, reuse=False, num_gpus=4):
    #inputs: [batch-size, 2048, 3]
    #=>      [batch-size, 3, 2048]
    begin = layers.base.transpose(inputs, (0, 2, 1), reuse=reuse, name='transpose')
    #        [batch-size, 3, 2048]
    #=>      [batch-size, 6,  512]
    core.summarize('inputs', begin, reuse=reuse)
    gpus = range(num_gpus)
    multiviews = []
    for view in range(views):
        with core.device('/gpu:{}'.format(view%num_gpus)):
            x = ops.projection_transform(begin, 32, reuse=reuse, name='projection_0-{}'.format(view), act='squash')
            core.summarize('projection_0_{}'.format(view), x, reuse=reuse)
            #x = permutation_transform(x, 9, reuse=reuse, name='projection_1-{}'.format(view), act='squash')
            #ops.core.summarize('projection_1_{}'.format(view), x, reuse=reuse)
            #x = permutation_transform(x, 32, reuse=reuse, name='projection_2-{}'.format(view), act='squash')
            #ops.core.summarize('projection_2_{}'.format(view), x, reuse=reuse)
            x = layers.capsules.dense(x, 32, 32, epsilon=1e-10, reuse=reuse, name='caps_fully_connected-{}'.format(view),
                                      act='squash', share_weights=True)
            core.summarize('caps_fully_connected-{}'.format(view), x, reuse=reuse)
            multiviews.append(x)
    with core.device('/gpu:0'):
        x = layers.merge.concat(multiviews, axis=2, reuse=reuse, name='concatenate')
        core.summarize('concatenate', x, reuse=reuse)
        x = layers.capsules.dense(x, nclass, 16, epsilon=1e-10, reuse=reuse, name='caps_fully_connected', act='squash')
        core.summarize('caps_fully_connected', x, reuse=reuse)
        x = layers.capsules.norm(x, axis=1, safe=True, epsilon=1e-10, name='norm', reuse=reuse)
        core.summarize('norm', x, reuse=reuse)
    return x
