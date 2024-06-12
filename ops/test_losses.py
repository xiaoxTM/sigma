import unittest
from . import losses, core
import numpy as np
from . import core
from .core import __tensorflow__ as stf
from numpy import linalg as la

def array2samples_distance(a1, a2):
    npoints, nfeatures = a1.shape
    a1_expand = np.tile(a1, (npoints, 1))
    a2_expand = np.reshape(
            np.tile(np.expand_dims(a2, 1),
                (1, npoints, 1)),
            (-1, nfeatures))
    dist = la.norm(a1_expand-a2_expand, axis=1)
    dist = np.reshape(dist, (npoints, npoints))
    dist = np.min(dist, axis=1)
    dist = np.mean(dist)
    return dist

def chamfer_distance_numpy(a1, a2):
    batchsize, npoints, nfeatures = a1.shape
    dist = 0
    for i in range(batchsize):
        ad1 = array2samples_distance(a1[i], a2[i])
        ad2 = array2samples_distance(a2[i], a1[i])
        dist = dist + (ad1 + ad2) / batchsize
    return dist / 2.0

class LossesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = stf.session()

    def test_chamfer_loss(self):
        batchsize = 8
        npoints = 20
        nfeatures = 4
        np.random.seed(1)
        inputs = np.random.randint(0, high=4, size=(batchsize, npoints, nfeatures))
        targets = np.random.randint(0, high=4, size=(batchsize, npoints, nfeatures))

        input_tf = core.constant(inputs, dtype=core.float64)
        target_tf = core.constant(targets, dtype=core.float64)

        with self.subTest(idx=0):
            numpy_out = chamfer_distance_numpy(inputs, targets)
            tensorflow_out = losses.chamfer_loss(axis=1)(input_tf, target_tf)
            tf_out = core.run(self.sess, tensorflow_out)
            self.assertTrue(np.abs(numpy_out-tf_out) < 0.0001)
        with self.subTest(idx=1):
            tensorflow_out = losses.chamfer_loss(axis=1)(input_tf, target_tf)
            tf_out = core.run(self.sess, tensorflow_out)
            tensorflow_out2 = losses.chamfer_loss(axis=1)(target_tf, input_tf)
            tf_out2 = core.run(self.sess, tensorflow_out2)
            self.assertTrue(np.abs(tf_out-tf_out2) < 0.00001)
