import unittest
from sigma.utils.utils import *

@deprecated('deprecated_func is deprecated, please use new func instead')
def deprecated_func():
    pass

class TestUtils(unittest.TestCase):
    def test_deprecated(self):
        self.assertWarns(DeprecationWarning, msg='deprecated_func is deprecated, please use new func instead')

    def test_shape_statistics(self):
        stat = shape_statistics([None, 1, 3, 2, 4, -1, None])
        self.assertListEqual(stat['None'], [0,6])
        self.assertListEqual(stat['-1'],[5])

    def test_intsize(self):
        self.assertEqual(intsize(200),3)
        self.assertEqual(intsize(-30),2)
        self.assertEqual(intsize(-30,True),3)