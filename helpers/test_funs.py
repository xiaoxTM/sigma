import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest

from . import funs
import numpy as np

@funs.typecheck(a=int, b=list)
def test(a, b, c):
    pass


class FunsTest(unittest.TestCase):
    def test_typecheck(self):
        with self.subTest():
            self.assertRaises(TypeError, lambda: test('3', [], 20))
        with self.subTest():
            self.assertRaises(TypeError, lambda: test(3, '3', 10))

    def test_hitmap(self):
        #predict=np.random.randint(0, 40, size=50)
        #label = np.random.randint(0, 40, size=50)
        result = [1,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1]
        with self.subTest():
            predict = np.asarray([0,3,2,3,1], dtype=np.int32)
            label   = np.asarray([0,2,3,3,2], dtype=np.int32)
            matrix = funs.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
        with self.subTest():
            predict = np.asarray([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,0,1],[0,1,0,0]], dtype=np.int32)
            label   = np.asarray([0,2,3,3,2], dtype=np.int32)
            matrix = funs.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
        with self.subTest():
            predict = np.asarray([0,3,2,3,1], dtype=np.int32)
            label   = np.asarray([[1,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0]], dtype=np.int32)
            matrix = funs.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
        with self.subTest():
            predict = np.asarray([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,0,1],[0,1,0,0]], dtype=np.int32)
            label   = np.asarray([[1,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,1],[0,0,1,0]], dtype=np.int32)
            matrix = funs.hitmap(predict, label, 4).astype(np.int32)
            self.assertListEqual(matrix.flatten().tolist(), result)
