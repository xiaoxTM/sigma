import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest
import mnist

class TestMnist(unittest.TestCase):

    def test_load_with_nclass(self):
        [xtrain, ytrain], [xvalid, yvalid] = mnist.load('/home/xiaox/studio/db/mnist', to_tensor=False, onehot=10, nclass=10)

        self.assertEqual(xtrain.shape[0], ytrain.shape[0])
        self.assertEqual(xtrain.shape[1:], (28, 28, 1))

        self.assertEqual(xvalid.shape[0], yvalid.shape[0])
        self.assertEqual(xvalid.shape[1:], (28, 28, 1))

        self.assertEqual(ytrain.shape[-1], 10)
        self.assertEqual(ytrain.shape[-1], yvalid.shape[-1])

    def test_load_no_nclass(self):
        [xtrain, ytrain], [xvalid, yvalid] = mnist.load('/home/xiaox/studio/db/mnist', to_tensor=False)

        self.assertEqual(xtrain.shape[0], ytrain.shape[0])
        self.assertEqual(xtrain.shape[1:], (28, 28, 1))

        self.assertEqual(xvalid.shape[0], yvalid.shape[0])
        self.assertEqual(xvalid.shape[1:], (28, 28, 1))

        self.assertEqual(len(ytrain.shape), 1)
        self.assertEqual(len(ytrain.shape), len(yvalid.shape))
