import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import sigma
from sigma import layers

def build():
    x = sigma.placeholder('float32', shape=(3,64,64,3))
    with sigma.defaults(layers.convs.conv2d, layers.pools.max_pool2d, padding='same', act='relu', pshape=(4,4)):
        x = layers.convs.conv2d(x, 10)
        x = layers.pools.max_pool2d(x)
        x = layers.convs.conv2d(x, 20)
        x = layers.convs.conv2d(x, 40)
        x = layers.pools.max_pool2d(x)
        x = layers.base.flatten(x)
        x = layers.convs.fully_conv(x, 10)
        return x

if __name__ == '__main__':
    build()
