import sigma
from .. import layers

def vgg16(input_shape, nclass=1000, classification=False,
          reuse=False, scope='vgg16'):
    with sigma.defaults(layers.convs.conv2d, stride=2, padding='same', act='relu'):
        x = layers.base.input_spec(input_shape, reuse=reuse, scope=scope)
        
        x = layers.convs.conv2d(x, 64, name='block1_conv1')
        x = layers.convs.conv2d(x, 64, name='block1_conv2')
        x = layers.pools.max_pool2d(x, 2, name='block1_pool')
        
        x = layers.convs.conv2d(x, 128, name='block2_conv1')
        x = layers.convs.conv2d(x, 128, name='block2_conv2')
        x = layers.pools.max_pool2d(x, 2, name='block2_pool')
        
        x = layers.convs.conv2d(x, 256, name='block3_conv1')
        x = layers.convs.conv2d(x, 256, name='block3_conv2')
        x = layers.convs.conv2d(x, 256, name='block3_conv3')
        x = layers.pools.max_pool2d(x, 2, name='block3_pool')
        
        x = layers.convs.conv2d(x, 512, name='block4_conv1')
        x = layers.convs.conv2d(x, 512, name='block4_conv2')
        x = layers.convs.conv2d(x, 512, name='block4_conv3')
        x = layers.pools.max_pool2d(x, 2, name='block4_pool')

        x = layers.convs.conv2d(x, 512, name='block5_conv1')
        x = layers.convs.conv2d(x, 512, name='block5_conv2')
        x = layers.convs.conv2d(x, 512, name='block5_conv3')
        x = layers.pools.max_pool2d(x, 2, name='block5_pool')
        
        if classification:
            x = layers.base.flatten(x, name='flatten')
            x = layers.convs.dense(x, 4096, name='fc1')
            x = layers.convs.dense(x, 4096, name='fc2')
            x = layers.convs.dense(x, nclass, name='predictions')

    return x



def vgg19(input_shape, nclass=1000, classification=False,
          reuse=False, scope='vgg19'):
    with sigma.defaults(layers.convs.conv2d, stride=2, padding='same', act='relu'):
        x = layers.base.input_spec(input_shape, reuse=reuse, scope=scope)
        
        x = layers.convs.conv2d(x, 64, name='block1_conv1')
        x = layers.convs.conv2d(x, 64, name='block1_conv2')
        x = layers.pools.max_pool2d(x, 2, name='block1_pool')
        
        x = layers.convs.conv2d(x, 128, name='block2_conv1')
        x = layers.convs.conv2d(x, 128, name='block2_conv2')
        x = layers.pools.max_pool2d(x, 2, name='block2_pool')
        
        x = layers.convs.conv2d(x, 256, name='block3_conv1')
        x = layers.convs.conv2d(x, 256, name='block3_conv2')
        x = layers.convs.conv2d(x, 256, name='block3_conv3')
        x = layers.convs.conv2d(x, 256, name='block3_conv4')
        x = layers.pools.max_pool2d(x, 2, name='block3_pool')
        
        x = layers.convs.conv2d(x, 512, name='block4_conv1')
        x = layers.convs.conv2d(x, 512, name='block4_conv2')
        x = layers.convs.conv2d(x, 512, name='block4_conv3')
        x = layers.convs.conv2d(x, 512, name='block4_conv4')
        x = layers.pools.max_pool2d(x, 2, name='block4_pool')

        x = layers.convs.conv2d(x, 512, name='block5_conv1')
        x = layers.convs.conv2d(x, 512, name='block5_conv2')
        x = layers.convs.conv2d(x, 512, name='block5_conv3')
        x = layers.convs.conv2d(x, 512, name='block5_conv4')
        x = layers.pools.max_pool2d(x, 2, name='block5_pool')
        
        if classification:
            x = layers.base.flatten(x, name='flatten')
            x = layers.convs.dense(x, 4096, name='fc1')
            x = layers.convs.dense(x, 4096, name='fc2')
            x = layers.convs.dense(x, nclass, name='predictions')

    return x