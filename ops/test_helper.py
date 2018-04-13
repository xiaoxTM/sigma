import unittest
from . import helper, core
import numpy as np

class HelperTest(unittest.TestCase):

    def test_name_space(self):
        self.assertEqual(helper.dispatch_name('game'), 'game-0')
        self.assertEqual(helper.dispatch_name('game'), 'game-1')
        self.assertEqual(helper.dispatch_name(None, -1), 'game-1')
        self.assertEqual(helper.dispatch_name('game'), 'game-2')
        self.assertEqual(helper.dispatch_name('game', 1), 'game-1')

        self.assertRaises(TypeError, lambda: helper.dispatch_name())
        self.assertRaises(TypeError, lambda: helper.dispatch_name(3.0))
        self.assertRaises(TypeError, lambda: helper.dispatch_name('game', 3.0))
        self.assertRaises(TypeError, lambda: helper.dispatch_name('game', '3.0'))
        self.assertRaises(ValueError, lambda: helper.dispatch_name('dota', -1))


    def test_assign_scope(self):
        self.assertRaises(ValueError, lambda: helper.assign_scope(None, 'test', None))
        _, name_with_ltype, name = helper.assign_scope('dense-1', 'train', 'dense')
        self.assertEqual(name_with_ltype, 'dense-1/dense')
        self.assertEqual(name, 'dense-1')

        _, name_with_ltype, name = helper.assign_scope(None, 'train', 'dense')
        self.assertEqual(name_with_ltype, 'dense-0/dense')
        self.assertEqual(name, 'dense-0')


    def test_depth(self):
        x = core.placeholder(dtype=core.float32, shape=(20,32,32,4))
        self.assertEqual(helper.depth(x), 4)

        x = core.placeholder(dtype=core.float32, shape=(20,32,4))
        self.assertEqual(helper.depth(x), 4)


    def test_name_normalize(self):
        self.assertRaises(TypeError, lambda: helper.name_normalize(3.0))
        self.assertEqual(helper.name_normalize('dense-1/dense/routing/logits-1'), 'dense-1')
        self.assertEqual(helper.name_normalize('test/dense-1/dense/routing/logits-1', 'test'), 'dense-1')


    def test_normalize_axes(self):
        self.assertRaises(TypeError, lambda: helper.normalize_axes('shape'))
        self.assertRaises(TypeError, lambda: helper.normalize_axes([3,5,6,2], 3.0))
        self.assertRaises(TypeError, lambda: helper.normalize_axes([3,5,6,2], '-1'))
        self.assertEqual(helper.normalize_axes([3,5,6,2]), 3)
        self.assertEqual(helper.normalize_axes([3,5,6,2], 1), 1)
        self.assertEqual(helper.normalize_axes([3,5,6,2], -2), 2)


    def test_get_output_shape(self):
        self.assertRaises(TypeError, lambda: helper.get_output_shape(2.0, 4, (2,2), (3,3), 'VALID'))
        self.assertRaises(TypeError, lambda: helper.get_output_shape((2.2), 4, 3.0, (3,3), 'SAME'))
        self.assertRaises(TypeError, lambda: helper.get_output_shape((2.2), 4, (3,3), 3.0, 'VALID'))

        self.assertRaises(ValueError, lambda: helper.get_output_shape((2,), 4, (3,3), (3,3), 'VALID'))
        self.assertRaises(ValueError, lambda: helper.get_output_shape((2,2), 4, (3,), (3,3), 'VALID'))
        self.assertRaises(ValueError, lambda: helper.get_output_shape((2,2), 4, (3,3), (3,), 'VALID'))
        self.assertRaises(ValueError, lambda: helper.get_output_shape((2,2), 4, (2,2), (2,2), 'game'))

        shape = helper.get_output_shape((10,32,32,3), 20, (1,3,3,1), (1,2,2,1), 'same')
        self.assertTrue(np.all(shape==[10,16,16,20]))
        shape = helper.get_output_shape((10,32,32,3), 20, (1,3,3,1), (1,2,2,1), 'valid')
        self.assertTrue(np.all(shape==[10,15,15,20]))


    def test_norm_input_1d(self):
        self.assertRaises(ValueError, lambda: helper.norm_input_1d((2,3)))
        self.assertRaises(TypeError, lambda: helper.norm_input_1d('2'))

        shape = helper.norm_input_1d(3)
        self.assertTrue(np.all(shape==[1,3,1]))
        shape = helper.norm_input_1d((3,))
        self.assertTrue(np.all(shape==[1,3,1]))
        shape = helper.norm_input_1d((1,2,3))
        self.assertTrue(np.all(shape==[1,2,3]))


    def test_norm_input_2d(self):
        self.assertRaises(ValueError, lambda: helper.norm_input_2d((2,3,2)))
        self.assertRaises(TypeError, lambda: helper.norm_input_2d('2'))

        shape = helper.norm_input_2d(3)
        self.assertTrue(np.all(shape==[1,3,3,1]))
        shape = helper.norm_input_2d((3,))
        self.assertTrue(np.all(shape==[1,3,3,1]))
        shape = helper.norm_input_2d((1,2,2,3))
        self.assertTrue(np.all(shape==[1,2,2,3]))


    def test_norm_input_3d(self):
        self.assertRaises(ValueError, lambda: helper.norm_input_3d((2,3,3,2)))
        self.assertRaises(TypeError, lambda: helper.norm_input_3d('2'))

        shape = helper.norm_input_3d(3)
        self.assertTrue(np.all(shape==[1,3,3,3,1]))
        shape = helper.norm_input_3d((3,))
        self.assertTrue(np.all(shape==[1,3,3,3,1]))
        shape = helper.norm_input_3d((1,2,2,3,3))
        self.assertTrue(np.all(shape==[1,2,2,3,3]))
