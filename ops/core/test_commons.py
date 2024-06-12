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
        with self.subTest(idx=0):
            self.assertEqual(len(stat['nones']), 2)
        with self.subTest(idx=1):
            self.assertEqual(stat['nones'][0], 0)
        with self.subTest(idx=2):
            self.assertEqual(stat['nones'][1], 5)
        with self.subTest(idx=3):
            self.assertEqual(len(stat['-1']), 1)
        with self.subTest(idx=4):
            self.assertEqual(stat['-1'][0], 4)


    def test_encode(self):
        with self.subTest(idx=5):
            self.assertRaises(TypeError, lambda: cms.encode(dict))
        with self.subTest(idx=6):
            self.assertRaises(TypeError, lambda: cms.encode(30))
        with self.subTest(idx=7):
            self.assertRaises(TypeError, lambda: cms.encode(12.0))


    def test_decode(self):
        with self.subTest(idx=8):
            self.assertRaises(TypeError, lambda: cms.decode(dict))
        with self.subTest(idx=9):
            self.assertRaises(TypeError, lambda: cms.decode(30))
        with self.subTest(idx=10):
            self.assertRaises(TypeError, lambda: cms.decode(12.0))


    # @classmethod
    # def tearDownClass(cls):
    #     stf.close_session(cls.sess)
