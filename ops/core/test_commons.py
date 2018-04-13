import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest
from . import commons as cms
import numpy as np

class CommonsTest(unittest.TestCase):
    # self.sess = None

    # @classmethod
    # def setUpClass(cls):
    #     cls.sess = stf.session()

    def test_shape_statistics(self):
        stat = cms.shape_statistics([None, 1, 3, 4, -1, None, 1])
        self.assertEqual(len(stat['nones']), 2)
        self.assertEqual(stat['nones'][0], 0)
        self.assertEqual(stat['nones'][1], 5)
        self.assertEqual(len(stat['-1']), 1)
        self.assertEqual(stat['-1'][0], 4)

    def test_encode(self):
        self.assertRaises(TypeError, lambda: cms.encode(dict))
        self.assertRaises(TypeError, lambda: cms.encode(30))
        self.assertRaises(TypeError, lambda: cms.encode(12.0))

    def test_decode(self):
        self.assertRaises(TypeError, lambda: cms.decode(dict))
        self.assertRaises(TypeError, lambda: cms.decode(30))
        self.assertRaises(TypeError, lambda: cms.decode(12.0))


    # @classmethod
    # def tearDownClass(cls):
    #     stf.close_session(cls.sess)
