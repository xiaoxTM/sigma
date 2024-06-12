import unittest
from ..ops import core
import numpy as np
from ..ops.core import __tensorflow__ as stf

from . import base

class BaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = stf.session()

    def test_flatten(self):
        batchsize = 8
        npoints = 20
        nfeatures = 4
        np.random.seed(1)
        inputs = np.random.randint(0, high=4, size=(batchsize, npoints, nfeatures))
        tfinputs = base.input_spec([batchsize, npoints, nfeatures], dtype=core.int32)
        np_flatten = np.reshape(inputs, [batchsize, -1])
        tf_flatten = base.flatten(tfinputs)
        tf_out = core.run(self.sess, tf_flatten, feed_dict={tfinputs:inputs})
        self.assertListEqual(np_flatten.tolist(), tf_out.tolist())

    def test_transpose(self):
        batchsize = 8
        npoints = 20
        nfeatures = 4
        np.random.seed(1)
        inputs = np.random.randint(0, high=4, size=(batchsize, npoints, nfeatures))
        tfinputs = base.input_spec([batchsize, npoints, nfeatures], dtype=core.int32)
        np_transpose = np.transpose(inputs, (0,2,1))
        trans = base.transpose(tfinputs, (0, 2, 1))
        tf_transpose = core.run(self.sess, trans, feed_dict={tfinputs:inputs})
        self.assertListEqual(np_transpose.flatten().tolist(), tf_transpose.flatten().tolist())


    #def test_maskout(self):
    #    batchsize = 8
    #    npoints = 20
    #    nfeatures = 4
    #    np.random.seed(1)
    #    inputs = np.random.randint(0, high=4, size=(batchsize, nfeatures, npoints))
    #    inputs = inputs.astype(np.float32)
    #    index = np.random.randint(0, 20, size=(batchsize,))
    #    tfinputs = base.input_spec([batchsize, nfeatures, npoints], dtype=core.float32)
    #    tf_index = base.input_spec([batchsize], dtype=core.int32)

    #    with self.subTest(idx=0):
    #        np_maskout_drop = inputs[:, :, index]
    #        tf_maskout_drop = base.maskout(tfinputs, tf_index, axis=-1, drop=True, onehot=False)
    #        tf_drop = core.run(self.sess, tf_maskout_drop, feed_dict={tfinputs:inputs, tf_index:index})
    #        self.assertListEqual(np_maskout_drop.flatten().tolist(), tf_drop.flatten().tolist())

    #    with self.subTest(idx=1):
    #        mask = np.ones_like(inputs)
    #        mask[:, 1, :] = 1
    #        np_maskout = np.reshape(inputs * mask, [batchsize, -1] )
    #        tf_maskout = base.maskout(tfinputs, 1, axis=1, onehot=False)
    #        tf_drop = core.run(self.sess, tf_maskout, feed_dict={tfinputs:inputs, tf_index:index})
    #        self.assertListEqual(np_maskout.flatten(), tf_drop.flatten())

    #    with self.subTest(idx=2):
    #        mask = np.ones_like(inputs)
    #        mask[:, 1, :] = 1
    #        np_maskout = inputs * mask
    #        tf_maskout = base.maskout(tfinputs, 1, axis=1, flatten=False, onehot=False)
    #        tf_drop = core.run(self.sess, tf_maskout, feed_dict={tfinputs:inputs, tf_index:index})
    #        self.assertListEqual(np_maskout.flatten(), tf_drop.flatten())
