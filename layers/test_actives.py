import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import ops
from . import actives
from numpy import linalg as la
import numpy as np
import unittest

class ActivesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = ops.core.session()


    def test_squash(self):
        data = np.asarray([[0.6, 0.8, -7.4]])
        data_norm = la.norm(data)
        data_square_norm = data_norm**2
        data_squash = (data_square_norm / (1 + data_square_norm)) * (data / data_norm)

        variable = ops.core.to_tensor(data)
        squash = actives.squash(variable, axis=1)

        _squash = ops.core.run(self.sess, squash)
        for ds, ss in zip(data_squash, _squash):
            for d,s in zip(ds, ss):
                self.assertAlmostEqual(d, s)


    @classmethod
    def tearDownClass(cls):
        ops.core.close_session(cls.sess)
