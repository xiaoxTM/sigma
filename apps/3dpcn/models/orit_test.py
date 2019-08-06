import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import ops, layers, engine

import numpy as np


if __name__ == '__main__':
    inputs = layers.base.input_spec([None, 10, 3])
    y = layers.convs.order_invariance_transform(inputs, 10, 3)

    sess, _, _, _ = engine.session();

    with sess:
        for i in range(10):
            x = np.random.randn(1, 10, 3)
            xs = []
            for j in range(5):
                np.random.shuffle(x)
                y_ = ops.core.run(sess, y, feed_dict={inputs:x})
                y_ = np.squeeze(y_)
                xs.append(y_)
            r = np.concatenate(xs, axis=1)
            np.savetxt('{}.txt'.format(i), r)
