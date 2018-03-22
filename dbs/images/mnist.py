"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import os.path
from sigma import helpers, ops, engine

def load(dirs=None, to_tensor=True, onehot=False, nclass=None):
    """ load mnist data from directory
        output in .tensor form
    """
    def _load(dset, nsamples):
        with open(os.path.join(dirs, '{}-images.idx3-ubyte'.format(dset)),
                  'rb') as fd:
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            x = loaded[16:].reshape(
              (nsamples, 28, 28, 1)).astype(np.float32) / 255.0
        with open(os.path.join(dirs, '{}-labels.idx1-ubyte'.format(dset)),
                  'rb') as fd:
            loaded = np.fromfile(file=fd, dtype=np.uint8)
            y = loaded[8:].reshape((nsamples)).astype(np.int32)
        if to_tensor:
            x = ops.core.to_tensor(x, ops.core.float32)
            if onehot:
                y = ops.core.one_hot(y, depth=10, axis=1, dtype=core.int32)
        else:
            if onehot:
                y = helpers.one_hot(y, nclass)
        return x, y
    if dirs is None:
        (xtrain, ytrain), (xvalid, yvalid) = helpers.ios.load_mnist('/tmp/mnist.npz')
    else:
        xtrain, ytrain = _load('train', 60000)
        xvalid, yvalid = _load('t10k', 10000)
    return [xtrain, ytrain], [xvalid, yvalid]


def sampler(dirs, is_training, batch_size,
            onehot=False,
            to_tensor=False,
            nclass=None,
            nthreads=None,
            capacity=None,
            min_after_dequeue=None,
            allow_samller_final_batch=False):
    train, valid = load(dirs, to_tensor, onehot, nclass)

    data = valid
    if is_training:
        data = train
    if to_tensor:
        data_queues = helpers.ios.slice_input_producer(data)
        x, y = ops.core.shuffle_batch(data_queues, num_threads=nthreads,
                                      batch_size=batch_size,
                                      capacity=capacity,
                                      min_after_dequeue=min_after_dequeue,
                                      allow_smaller_final_batch=allow_smaller_final_batch)
        return (x, y)
    else:
        x, y = data
        index = np.arange(x.shape[0], dtype=np.int32)
        while True:
            np.random.shuffle(index)
            # get the previous `batch-size` samples
            index = index[:batch_size]
            if onehot:
                yield [x[index, :].reshape((-1, 28, 28, 1)), y[index]]
            else:
                yield [x[index, :].reshape((-1, 28, 28, 1)), y[index].reshape((-1, 1))]
