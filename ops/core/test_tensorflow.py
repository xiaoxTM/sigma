import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest
from . import __tensorflow__ as stf
import numpy as np
from numpy import linalg as la

class TensorFlowTest(unittest.TestCase):
    # self.sess = None

    @classmethod
    def setUpClass(cls):
        cls.sess = stf.session()


    def test_range(self):
        arange = stf.range(10, dtype=stf.int32)
        _arange = stf.run(self.sess, arange)
        self.assertTrue(np.all(_arange==np.arange(10, dtype=np.int32)))
        arange = stf.range(10, 100, 1, dtype=stf.int32)
        _arange = stf.run(self.sess, arange)
        self.assertTrue(np.all(_arange==np.arange(10, 100, 1, dtype=np.int32)))


    def test_norm(self):
        data = np.asarray([[0.6, 0.8, -7.4]])
        data_norm = la.norm(data)

        x = stf.norm(data)
        _x = stf.run(self.sess, x)
        self.assertTrue(np.all(data_norm==_x))


    def test_reshape(self):
        x = stf.placeholder(dtype=stf.float32, shape=(3, 4, 5, 6))
        x_shaped  = stf.reshape(x, (3, -1, 10, 3), True)
        _shape = np.asarray(stf.shape(x_shaped, True))
        # _shape = stf.run(self.sess, shape)
        self.assertTrue(np.all(_shape == np.asarray((3, 4, 10, 3))))

        self.assertRaises(ValueError, lambda: stf.reshape(x, (3, -1, -1, 6), True))
        self.assertRaises(ValueError, lambda: stf.reshape(x, (3, None, -1, 6), True))
        self.assertRaises(ValueError, lambda: stf.reshape(x, (3, None, None, 6), True))


    def test_max(self):
        x = [1, 4, 3, 9, 6]
        m = stf.max(x)
        _m = stf.run(self.sess, m)
        self.assertEqual(_m, 9)

        m = stf.max(x, [2, 5, 6, 7, 8])
        _m = stf.run(self.sess, m)
        self.assertTrue(np.all(_m==np.asarray([2, 5, 6, 9, 8])))


    def test_min(self):
        x = [1, 4, 3, 9, 6]
        m = stf.min(x)
        _m = stf.run(self.sess, m)
        self.assertEqual(_m, 1)

        m = stf.min(x, [2, 5, 6, 7, 8])
        _m = stf.run(self.sess, m)
        self.assertTrue(np.all(_m==np.asarray([1, 4, 3, 7, 6])))


    @classmethod
    def tearDownClass(cls):
        stf.close_session(cls.sess)
