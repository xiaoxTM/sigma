
import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma import ops, layers, status
import numpy as np
import tensorflow as tf
import unittest

class NormalizationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = ops.core.session()


    def test_batch_norm(self):
        batchsize = 2
        rows = 3
        cols = 3
        channels = 2
        x = layers.base.input_spec((batchsize, rows, cols, channels))
        nx = np.random.rand(batchsize, rows, cols, channels)

        with self.subTest(idx=0):
            sigma_x = layers.norms.batch_norm(x, epsilon=0.001, fused=True, is_training=True)
            batch_norm = tf.keras.layers.BatchNormalization(fused=True)
            tf_x = batch_norm(x, training=True)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

        with self.subTest(idx=1):
            sigma_x = layers.norms.batch_norm(x, epsilon=0.001, fused=False, is_training=True)
            batch_norm = tf.keras.layers.BatchNormalization(fused=False)
            tf_x = batch_norm(x, training=True)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

        with self.subTest(idx=2):
            def _batch_norm(reuse, is_training=True):
                return layers.norms.batch_norm(x, epsilon=0.001, fused=True, name='batch_norm', reuse=reuse, is_training=is_training)

            update_ops = ops.core.get_collection(ops.core.Collections.update_ops)
            sigma_x = _batch_norm(reuse=False)
            batch_norm = tf.keras.layers.BatchNormalization(fused=True)
            tf_x = batch_norm(x, training=True)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x,_ = ops.core.run(self.sess, [sigma_x, tf_x, update_ops], {x:nx})
            tf_x = batch_norm(x, training=False)
            nnx = np.random.rand(batchsize, rows, cols, channels)
            sigma_x = _batch_norm(reuse=True, is_training=False)
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nnx})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())



    #def test_batch_norm(self):
    #    data = np.asarray([[0.6, 0.8, -7.4]])
    #    data_norm = la.norm(data)
    #    data_square_norm = data_norm**2
    #    data_squash = (data_square_norm / (1 + data_square_norm)) * (data / data_norm)

    #    variable = ops.core.to_tensor(data)
    #    squash = actives.squash(variable)

    #    _squash = ops.core.run(self.sess, squash)
    #    self.assertListEqual(data_squash.tolist(), _squash.tolist())


    @classmethod
    def tearDownClass(cls):
        ops.core.close_session(cls.sess)
