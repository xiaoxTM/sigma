import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest
from . import cifar

class TestCifar(unittest.TestCase):

    def test_load_with_nclass(self):
        [xtrain, ytrain], [xvalid, yvalid] = cifar.load('/home/xiaox/studio/db/cifar/cifar-100-python', to_tensor=False, onehot=10, nclass=20)

        with self.subTest(idx=0):
            self.assertEqual(xtrain.shape[0], ytrain.shape[0])
        with self.subTest(idx=1):
            self.assertEqual(xtrain.shape[1:], (32, 32, 3))

        with self.subTest(idx=2):
            self.assertEqual(xvalid.shape[0], yvalid.shape[0])
        with self.subTest(idx=3):
            self.assertEqual(xvalid.shape[1:], (32, 32, 3))

        with self.subTest(idx=4):
            self.assertEqual(ytrain.shape[-1], 20)
        with self.subTest(idx=5):
            self.assertEqual(ytrain.shape[-1], yvalid.shape[-1])

    def test_load_no_nclass(self):
        [xtrain, ytrain], [xvalid, yvalid] = cifar.load('/home/xiaox/studio/db/cifar/cifar-100-python', to_tensor=False)

        with self.subTest(idx=6):
            self.assertEqual(xtrain.shape[0], ytrain.shape[0])
        with self.subTest(idx=7):
            self.assertEqual(xtrain.shape[1:], (32, 32, 3))

        with self.subTest(idx=8):
            self.assertEqual(xvalid.shape[0], yvalid.shape[0])
        with self.subTest(idx=9):
            self.assertEqual(xvalid.shape[1:], (32, 32, 3))

        with self.subTest(idx=10):
            self.assertEqual(len(ytrain.shape), 1)
        with self.subTest(idx=11):
            self.assertEqual(len(ytrain.shape), len(yvalid.shape))
