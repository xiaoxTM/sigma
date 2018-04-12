import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest
import cifar

class TestCifar(unittest.TestCase):

    def test_load_with_nclass(self):
        [xtrain, ytrain], [xvalid, yvalid] = cifar.load('/home/xiaox/studio/db/cifar/cifar-100-python', to_tensor=False, onehot=10, nclass=20)

        self.assertEqual(xtrain.shape[0], ytrain.shape[0])
        self.assertEqual(xtrain.shape[1:], (32, 32, 3))

        self.assertEqual(xvalid.shape[0], yvalid.shape[0])
        self.assertEqual(xvalid.shape[1:], (32, 32, 3))

        self.assertEqual(ytrain.shape[-1], 20)
        self.assertEqual(ytrain.shape[-1], yvalid.shape[-1])

    def test_load_no_nclass(self):
        [xtrain, ytrain], [xvalid, yvalid] = cifar.load('/home/xiaox/studio/db/cifar/cifar-100-python', to_tensor=False)

        self.assertEqual(xtrain.shape[0], ytrain.shape[0])
        self.assertEqual(xtrain.shape[1:], (32, 32, 3))

        self.assertEqual(xvalid.shape[0], yvalid.shape[0])
        self.assertEqual(xvalid.shape[1:], (32, 32, 3))

        self.assertEqual(len(ytrain.shape), 1)
        self.assertEqual(len(ytrain.shape), len(yvalid.shape))
