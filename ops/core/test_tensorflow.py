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
        with self.subTest(idx=0):
            self.assertListEqual(_arange.tolist(), list(range(10)))
        arange = stf.range(10, 100, 1, dtype=stf.int32)
        _arange = stf.run(self.sess, arange)
        with self.subTest(idx=1):
            self.assertListEqual(_arange.tolist(), list(range(10, 100, 1)))


    def test_norm(self):
        data = np.asarray([[0.6, 0.8, -7.4]])
        data_norm = la.norm(data)
        data_tf = stf.placeholder(shape=(1,3), dtype=stf.float32)
        x = stf.norm(data_tf,axis=1)
        _x = stf.run(self.sess, x, feed_dict={data_tf:data})
        with self.subTest(idx=2):
            self.assertAlmostEqual(data_norm, _x)


    def test_reshape(self):
        x = stf.placeholder(dtype=stf.float32, shape=(3, 4, 5, 6))
        x_shaped  = stf.reshape(x, (3, -1, 10, 3), True)
        _shape = stf.shape(x_shaped, True)
        # _shape = stf.run(self.sess, shape)
        with self.subTest(idx=3):
            self.assertListEqual(_shape, [3, 4, 10, 3])
        with self.subTest(idx=4):
            self.assertRaises(ValueError, lambda: stf.reshape(x, (3, -1, -1, 6), True))
        with self.subTest(idx=5):
            self.assertRaises(ValueError, lambda: stf.reshape(x, (3, None, -1, 6), True))
        with self.subTest(idx=6):
            self.assertRaises(ValueError, lambda: stf.reshape(x, (3, None, None, 6), True))

    #def test_flatten(self):
    #    x = stf.placeholder(dtype=stf.int32, shape=(2,3,4))
    #    x1 = stf.flatten(x)
    #    y = np.random.randint(0, size=(2,3,4))
    #    _y1 = y.flatten()
    #    _x1 = stf.run(self.sess, x1, feed_dict={x:y})
    #    self.assertTrue(np.all(_x1 == _y1))


    def test_max(self):
        x = [1, 4, 3, 9, 6]
        m = stf.max(x)
        _m = stf.run(self.sess, m)
        with self.subTest(idx=7):
            self.assertEqual(_m, 9)

        m = stf.max(x, [2, 5, 6, 7, 8])
        _m = stf.run(self.sess, m)
        with self.subTest(idx=8):
            self.assertListEqual(_m.tolist(), [2, 5, 6, 9, 8])


    def test_min(self):
        x = [1, 4, 3, 9, 6]
        m = stf.min(x)
        _m = stf.run(self.sess, m)
        with self.subTest(idx=9):
            self.assertEqual(_m, 1)

        m = stf.min(x, [2, 5, 6, 7, 8])
        _m = stf.run(self.sess, m)
        with self.subTest(idx=10):
            self.assertListEqual(_m.tolist(), [1, 4, 3, 7, 6])


    @classmethod
    def tearDownClass(cls):
        stf.close_session(cls.sess)
