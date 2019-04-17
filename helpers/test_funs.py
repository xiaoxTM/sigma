import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest

from . import funs


@funs.typecheck(a=int, b=list)
def test(a, b, c):
    pass


class FunsTest(unittest.TestCase):
    def test_typecheck(self):
        with self.subTest():
            self.assertRaises(TypeError, lambda: test('3', [], 20))
        with self.subTest():
            self.assertRaises(TypeError, lambda: test(3, '3', 10))
