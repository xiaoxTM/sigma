import unittest
from . import helper, core
import numpy as np

class HelperTest(unittest.TestCase):

    def test_name_space(self):
        with self.subTest(idx=0):
            self.assertEqual(helper.dispatch_name('game'), '.game-0.')
        with self.subTest(idx=1):
            self.assertEqual(helper.dispatch_name('game'), '.game-1.')
        with self.subTest(idx=2):
            self.assertEqual(helper.dispatch_name(None, -1), '.game-1.')
        with self.subTest(idx=3):
            self.assertEqual(helper.dispatch_name('game'), '.game-2.')
        with self.subTest(idx=4):
            self.assertEqual(helper.dispatch_name('game', 1), '.game-1.')

        with self.subTest(idx=5):
            self.assertRaises(TypeError, lambda: helper.dispatch_name())
        with self.subTest(idx=6):
            self.assertRaises(TypeError, lambda: helper.dispatch_name(3.0))
        with self.subTest(idx=7):
            self.assertRaises(TypeError, lambda: helper.dispatch_name('game', 3.0))
        with self.subTest(idx=8):
            self.assertRaises(TypeError, lambda: helper.dispatch_name('game', '3.0'))
        with self.subTest(idx=9):
            self.assertRaises(ValueError, lambda: helper.dispatch_name('dota', -1))


    def test_assign_scope(self):
        with self.subTest(idx=10):
            self.assertRaises(ValueError, lambda: helper.assign_scope(None, 'test', None))
        _, name_with_ltype, name = helper.assign_scope('dense-1', 'train', 'dense')
        with self.subTest(idx=11):
            self.assertEqual(name_with_ltype, '.dense-1./dense')
        with self.subTest(idx=12):
            self.assertEqual(name, '.dense-1.')

        _, name_with_ltype, name = helper.assign_scope(None, 'train', 'dense')
        with self.subTest(idx=13):
            self.assertEqual(name_with_ltype, '.dense-0./dense')
        with self.subTest(idx=14):
            self.assertEqual(name, '.dense-0.')


    def test_depth(self):
        x = core.placeholder(dtype=core.float32, shape=(20,32,32,4))
        with self.subTest(idx=15):
            self.assertEqual(helper.depth(x), 4)

        x = core.placeholder(dtype=core.float32, shape=(20,32,4))
        with self.subTest(idx=16):
            self.assertEqual(helper.depth(x), 4)


    def test_split_name(self):
        with self.subTest(idx=17):
            self.assertRaises(TypeError, lambda: helper.split_name(3.0))
        with self.subTest(idx=18):
            self.assertEqual(helper.split_name('dense/dense/.routing./logits:0', False), ['dense/dense', 'routing'])
        with self.subTest(idx=19):
            self.assertEqual(helper.split_name('test/dense/.dense./routing/logits:0'), 'dense')


    def test_normalize_axes(self):
        with self.subTest(idx=20):
            self.assertRaises(TypeError, lambda: helper.normalize_axes('shape'))
        with self.subTest(idx=21):
            self.assertRaises(TypeError, lambda: helper.normalize_axes([3,5,6,2], 3.0))
        with self.subTest(idx=22):
            self.assertRaises(TypeError, lambda: helper.normalize_axes([3,5,6,2], '-1'))
        with self.subTest(idx=23):
            self.assertEqual(helper.normalize_axes([3,5,6,2]), 3)
        with self.subTest(idx=24):
            self.assertEqual(helper.normalize_axes([3,5,6,2], 1), 1)
        with self.subTest(idx=25):
            self.assertEqual(helper.normalize_axes([3,5,6,2], -2), 2)


    def test_get_output_shape(self):
        with self.subTest(idx=26):
            self.assertRaises(TypeError, lambda: helper.get_output_shape(2.0, 4, (2,2), (3,3), 'VALID'))
        with self.subTest(idx=27):
            self.assertRaises(TypeError, lambda: helper.get_output_shape((2.2), 4, 3.0, (3,3), 'SAME'))
        with self.subTest(idx=28):
            self.assertRaises(TypeError, lambda: helper.get_output_shape((2.2), 4, (3,3), 3.0, 'VALID'))

        with self.subTest(idx=29):
            self.assertRaises(ValueError, lambda: helper.get_output_shape((2,), 4, (3,3), (3,3), 'VALID'))
        with self.subTest(idx=30):
            self.assertRaises(ValueError, lambda: helper.get_output_shape((2,2), 4, (3,), (3,3), 'VALID'))
        with self.subTest(idx=31):
            self.assertRaises(ValueError, lambda: helper.get_output_shape((2,2), 4, (3,3), (3,), 'VALID'))
        with self.subTest(idx=32):
            self.assertRaises(ValueError, lambda: helper.get_output_shape((2,2), 4, (2,2), (2,2), 'game'))

        shape = helper.get_output_shape((10,32,32,3), 20, (1,3,3,1), (1,2,2,1), 'same')
        with self.subTest(idx=33):
            self.assertListEqual(shape, [10,16,16,20])
        shape = helper.get_output_shape((10,32,32,3), 20, (1,3,3,1), (1,2,2,1), 'valid')
        with self.subTest(idx=34):
            self.assertListEqual(shape, [10,15,15,20])


    def test_norm_input_1d(self):
        with self.subTest(idx=35):
            self.assertRaises(ValueError, lambda: helper.norm_input_1d((2,3)))
        with self.subTest(idx=36):
            self.assertRaises(TypeError, lambda: helper.norm_input_1d('2'))

        shape = helper.norm_input_1d(3)
        with self.subTest(idx=37):
            self.assertListEqual(shape, [1,3,1])
        shape = helper.norm_input_1d((3,))
        with self.subTest(idx=38):
            self.assertListEqual(shape, [1,3,1])
        shape = helper.norm_input_1d((1,2,3))
        with self.subTest(idx=39):
            self.assertListEqual(shape, [1,2,3])


    def test_norm_input_2d(self):
        with self.subTest(idx=40):
            self.assertRaises(ValueError, lambda: helper.norm_input_2d((2,3,2)))
        with self.subTest(idx=41):
            self.assertRaises(TypeError, lambda: helper.norm_input_2d('2'))

        shape = helper.norm_input_2d(3)
        with self.subTest(idx=42):
            self.assertListEqual(shape, [1,3,3,1])
        shape = helper.norm_input_2d((3,))
        with self.subTest(idx=43):
            self.assertListEqual(shape, [1,3,3,1])
        shape = helper.norm_input_2d((1,2,2,3))
        with self.subTest(idx=44):
            self.assertListEqual(shape, [1,2,2,3])


    def test_norm_input_3d(self):
        with self.subTest(idx=45):
            self.assertRaises(ValueError, lambda: helper.norm_input_3d((2,3,3,2)))
        with self.subTest(idx=46):
            self.assertRaises(TypeError, lambda: helper.norm_input_3d('2'))

        shape = helper.norm_input_3d(3)
        with self.subTest(idx=47):
            self.assertListEqual(shape, [1,3,3,3,1])
        shape = helper.norm_input_3d((3,))
        with self.subTest(idx=48):
            self.assertListEqual(shape, [1,3,3,3,1])
        shape = helper.norm_input_3d((1,2,2,3,3))
        with self.subTest(idx=49):
            self.assertListEqual(shape, [1,2,2,3,3])
