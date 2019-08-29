import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import layers
from sigma.ops import core
from . import ops

def build_net(inputs, labels, loss='margin_loss', nclass=40, num_gpus=4, views=4, reuse=False, is_training=True, **kwargs):
    #inputs: [batch-size, 2048, 3]
    #=>      [batch-size, 3, 2048]
    begin = layers.base.transpose(inputs, (0, 2, 1), reuse=reuse, name='transpose')
    #        [batch-size, 3, 2048]
    core.summarize('inputs', begin, reuse=reuse)
    gpus = range(num_gpus)
    multiviews = []
    channels = nclass#int(2048 / views)
    for view in range(views):
        with core.device('/gpu:{}'.format(view%num_gpus)):
            #=> [batch-size, 3, 2048]
            #=> [batch-size, 12, 512]
            x = ops.projection_transform(begin, 12, reuse=reuse, name='projection_0-{}'.format(view), act='squash')
            core.summarize('projection_0_{}'.format(view), x, reuse=reuse)
            x = layers.norms.batch_norm(x, name='batch_norm_0-{}'.format(view), reuse=reuse, is_training=is_training)
            core.summarize('batch_norm_0-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 12, 512]
            #=> [batch-size, 24, 256]
            x = ops.projection_transform(x, 24, reuse=reuse, name='projection_1-{}'.format(view), act='squash')
            ops.core.summarize('projection_1_{}'.format(view), x, reuse=reuse)
            x = layers.norms.batch_norm(x, name='batch_norm_1-{}'.format(view), reuse=reuse, is_training=is_training)
            core.summarize('batch_norm_1-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 24, 256]
            #=> [batch-size, 48, 128]
            #x = ops.projection_transform(x, 48, reuse=reuse, name='projection_2-{}'.format(view), act='squash')
            #ops.core.summarize('projection_2_{}'.format(view), x, reuse=reuse)
            #x = layers.norms.batch_norm(x, name='batch_norm_2-{}'.format(view), reuse=reuse, is_training=is_training)
            #core.summarize('batch_norm_2-{}'.format(view), x, reuse=reuse)
            #=> [batch-size, 48, 128]
            #=> [batch-size, 32, channels]
            x = ops.permutation_transform(x, channels, 48, name='permutation_transform-{}'.format(view), act='squash', reuse=reuse)
            core.summarize('permutation_transform-{}'.format(view), x, reuse=reuse)
            multiviews.append(x)
    with core.device('/gpu:{}'.format(num_gpus-1)):
        #=> [batch-size, 48, channels]
        #=> [batch-size, 48, channels*views]
        x = layers.merge.concat(multiviews, axis=2, reuse=reuse, name='concatenate')
        core.summarize('concatenate', x, reuse=reuse)
        #x = layers.norms.batch_norm(x, name='batch_norm_3', reuse=reuse, is_training=is_training)
        #core.summarize('batch_norm_3', x, reuse=reuse)
        #=> [batch-size, 48, channels*views]
        #=> [batch-size, 16, nclass]
        x = layers.capsules.dense(x, nclass, 16, epsilon=1e-10, reuse=reuse, name='caps_fully_connected', act='squash')
        core.summarize('caps_fully_connected', x, reuse=reuse)
        # for reconstruction
        # for recognition
        #=> [batch-size, 16, nclass]
        #=> [batch-size, nclass] (recog)
        recog = layers.capsules.norm(x, axis=1, safe=True, epsilon=1e-10, name='norm', reuse=reuse)
        core.summarize('norm', recog, reuse=reuse)
 
        recog_loss_op = layers.losses.get(loss, recog, labels)
        ops.core.summarize('recognition-loss', recog_loss_op, 'scalar')
        metric_op = layers.metrics.accuracy([recog, labels])

        #=> [batch-size, 16, nclass]
        #=> [batch-size, 16 * nclass]
        if is_training:
            recon = layers.base.maskout(x, labels, name='maskout_train', reuse=reuse)
        else:
            recon = layers.base.maskout(x, name='maskout_valid', reuse=reuse)
        x = layers.convs.dense(recon, 256, name='fully_connected-1', reuse=reuse)
        x = layers.convs.dense(x, 512, name='fully_connected-2', reuse=reuse)
        x = layers.convs.dense(x, 1024, name='fully_connected-3', reuse=reuse)
        x = layers.convs.dense(x, 2048, name='fully_connected-4', reuse=reuse)
        x = layers.convs.dense(x, 2048 * 3, name='fully_connected-5', reuse=reuse)
        x = layers.base.reshape(x, [-1, 3, 2048], name='reshape', reuse=reuse) 
        recon_loss_op = layers.losses.mse([x, begin])
        ops.core.summarize('reconstruction-loss', recon_loss_op, 'scalar')
        loss_op = layers.math.add([recog_loss_op, recon_loss_op], [1, 3.072], name='add', reuse=reuse)
        ops.core.summarize('total-loss', loss_op, 'scalar')
    return loss_op, metric_op
