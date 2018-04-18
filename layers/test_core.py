import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import unittest
import sigma
import functools
from . import core

@core.defaultable
def make_test1(a, b=4, c='test1'):
    return (a, b, c)


@core.defaultable
def setb(b=30):
    return b


@core.defaultable
def make_test2(b=5, c='test2'):
    b = setb()
    return (b, c)


@core.defaultable
def make_test3(b=5, c='test2'):
    return (b, c)


def make_test4(a, b=0.1, c='test4'):
    x, y, z = make_test1(a, b, c)
    return (x, setb(), z)


class CoreTest(unittest.TestCase):

    def test_split_inputs(self):
        x = [3,4]
        x, y = core.split_inputs(x)
        with self.subTest(idx=0):
            self.assertEqual(x, 3)
        with self.subTest(idx=1):
            self.assertEqual(y, 4)

        x = [5]
        x, y = core.split_inputs(x)
        with self.subTest(idx=2):
            self.assertEqual(x, 5)
        with self.subTest(idx=3):
            self.assertEqual(y, None)

        with self.subTest(idx=4):
            self.assertRaises(TypeError, lambda: core.split_inputs('test'))
        with self.subTest(idx=5):
            self.assertRaises(ValueError, lambda: core.split_inputs([2,3,4]))


    def test_set_print(self):
        with self.subTest(idx=6):
            self.assertRaises(TypeError, lambda: core.set_print(mode=3))
        with self.subTest(idx=7):
            self.assertRaises(TypeError, lambda: core.set_print(True, 3))
        with self.subTest(idx=8):
            self.assertEqual(core.__graph__, True)
        core.set_print(False, True)
        with self.subTest(idx=9):
            self.assertEqual(core.__graph__, False)
        with self.subTest(idx=10):
            self.assertEqual(core.__details__, True)


    def test_set_defaults(self):
        with self.subTest(idx=11):
            self.assertRaises(TypeError, lambda: core.set_defaults('defaults'))
        with self.subTest(idx=12):
            self.assertRaises(TypeError, lambda: core.set_defaults(3))
        with self.subTest(idx=13):
            self.assertRaises(KeyError, lambda: core.set_defaults({'default':3}))
        core.set_defaults({'padding':'same'})
        with self.subTest(idx=14):
            self.assertEqual(core.__defaults__['padding'], 'same')


    def test_defaults(self):
        with core.defaults(make_test1, setb, b=6, c='test'):
            x, y, z = make_test1(a=100)
            with self.subTest(idx=15):
                self.assertEqual(x, 100)
            with self.subTest(idx=16):
                self.assertEqual(y, 6)
            with self.subTest(idx=17):
                self.assertEqual(z, 'test')
            m, n = make_test2(b=200)
            with self.subTest(idx=18):
                self.assertEqual(m, 6)
            with self.subTest(idx=19):
                self.assertEqual(n, 'test2')
            m, n = make_test3()
            with self.subTest(idx=20):
                self.assertEqual(m, 5)
        with core.defaults(b=101):
            x, y, z = make_test4(a=300)
            with self.subTest():
                self.assertEqual(x, 300)
            with self.subTest():
                self.assertEqual(y, 101)
            with self.subTest():
                self.assertEqual(z, 'test4')

        with core.defaults(c='test41'):
            x, y, z = make_test4(a=10)
            with self.subTest():
                self.assertEqual(y, 30)
            with self.subTest():
                self.assertEqual(z, 'test4')
