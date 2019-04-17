import sigma
from .. import layers


def _identity_block(inputs, ksize, nouts, stage, block):
    conv_name_base = 'res{}{}_branch'.format(stage, block)
    bn_name_base = 'bn{}{}_branch'.format(stage, block)
    
    x = layers.convs.conv2d(inputs, nouts[0], kshape=1, padding='valid',
                            name='{}2a'.format(conv_name_base))
    x = layers.norms.batch_norm(x, act='relu', name='{}2a'.format(bn_name_base))
    
    x = layers.convs.conv2d(x, nouts[1], kshape=ksize, padding='same',
                            name='{}2b'.format(conv_name_base))
    x = layers.norms.batch_norm(x, act='relu', name='{}2b'.format(bn_name_base))
    
    x = layers.convs.conv2d(x, nouts[2], kshape=1, padding='valid',
                            name='{}2c'.format(conv_name_base))
    x = layers.norms.batch_norm(x, name='{}2c'.format(bn_name_base))
    
    x = layers.merge.add([x, inputs])
    x = layers.actives.relu(x)
    
    return x


def _conv_block(inputs, ksize, nouts, stage, block, strides=(2, 2)):
    conv_name_base = 'res{}{}_branch'.format(stage, block)
    bn_name_base = 'bn{}{}_branch'.format(stage, block)
    
    x = layers.convs.conv2d(inputs, nouts[0], kshape=1, stride=strides,
                            padding='valid', name='{}2a'.format())
    x = layers.norms.batch_norm(x, act='relu', name='{}2a'.format(bn_name_base))
    
    x = layers.convs.conv2d(x, nouts[1], kshape=ksize, padding='same',
                            name='{}2b'.format(conv_name_base))
    x = layers.norms.batch_norm(x, act='relu', name='{}2b'.format(bn_name_base))
    
    x = layers.convs.conv2d(x, nouts[2], kshape=1, padding='valid',
                            name='{}2c'.format(conv_name_base))
    x = layers.norms.batch_norm(x, name='{}2c'.format(bn_name_base))
    
    shortcut = layers.convs.conv2d(inputs, nouts[2], stride=strides,
                                   padding='valid', name='{}1'.format(conv_name_base))
    shortcut = layers.norms.batch_norm(shortcut, name='{}1'.format(bn_name_base))
    
    x = layers.merge.add([x, shortcut])
    x = layers.actives.relu(x)
    
    return x


def resnet50(input_shape, nclass=1000, classification=False,
             reuse=False, scope='resnet50'):
    with sigma.defaults(reuse=reuse, scope=scope):
        x = layers.base.input_spec(input_shape)

        x = layers.convs.conv2d(x, 64, kshape=7, stride=2, padding='same', name='conv1')
        x = layers.norms.batch_norm(x, act='relu', name='bn_conv1')
        x = layers.pools.max_pool2d(x, pshape=3, stride=2)

        x = _conv_block(x, 3, [64, 64, 256], stage=2, block='a', stride=1)
        x = _identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = _identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = _conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = _identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = _identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = _identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = _conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = _identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = _conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = _identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = layers.pools.avg_pool2d(x, 7, name='avg_pool')

        if classification:
            x = layers.base.flatten(x)
            x = layers.convs.dense(x, nclass, act='softmax', name='fc1000')
    return x