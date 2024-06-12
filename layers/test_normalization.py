
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
        self.maxDiff = None

        with self.subTest(idx=0):
            # train=True with fused=True
            sigma_x = layers.norms.batch_norm(x, epsilon=0.001, fused=True)
            batch_norm = tf.keras.layers.BatchNormalization(fused=True)
            tf_x = batch_norm(x, training=True)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx, status.is_training:True})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

        with self.subTest(idx=1):
            # train=True with fused=False
            sigma_x = layers.norms.batch_norm(x, epsilon=0.001, fused=False)
            batch_norm = tf.keras.layers.BatchNormalization(fused=False)
            tf_x = batch_norm(x, training=True)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx, status.is_training:True})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

        # with self.subTest(idx=2):
        #     # train=True then train=False with Fused=True
        #     def _batch_norm(reuse):
        #         return layers.norms.batch_norm(x, epsilon=0.001, fused=True, name='batch_norm', reuse=reuse)
        #     sigma_x = _batch_norm(reuse=False)
        #     batch_norm = tf.keras.layers.BatchNormalization(fused=True)
        #     tf_x_train = batch_norm(x, training=True)
        #     # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, batch_norm.updates)
        #     with tf.control_dependencies(batch_norm.updates):
        #         tf_x_train = tf.identity(tf_x_train)
        #     ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
        #     _sigma_x, _tf_x, = ops.core.run(self.sess, [sigma_x, tf_x_train], {x:nx, status.is_training:True})
        #     batch_norm.moving_mean = ops.core.runtime_print(batch_norm.moving_mean, 'keras mm update')
        #     self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())
        #     # update the moving_mean and moving_variance
        #     tf_x = batch_norm(x, training=False)
        #     nnx = np.random.rand(batchsize, rows, cols, channels)
        #     _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nnx, status.is_training:False})
        #     self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())
        #
        # with self.subTest(idx=3):
        #     # train=True then train=False with Fused=False
        #     def _batch_norm(reuse):
        #         return layers.norms.batch_norm(x, epsilon=0.001, fused=False, name='batch_norm_1', reuse=reuse)
        #     sigma_x = _batch_norm(reuse=False)
        #     batch_norm = tf.keras.layers.BatchNormalization(fused=False)
        #     tf_x = batch_norm(x, training=True)
        #     ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
        #     _sigma_x, _tf_x, = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx, status.is_training:True})
        #     self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())
        #
        #     tf_x = batch_norm(x, training=False)
        #     nnx = np.random.rand(batchsize, rows, cols, channels)
        #     _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nnx, status.is_training:False})
        #     self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

        with self.subTest(idx=4):
            # train=False with fused=True
            sigma_x = layers.norms.batch_norm(x, epsilon=0.001, fused=True)
            batch_norm = tf.keras.layers.BatchNormalization(fused=True)
            tf_x = batch_norm(x, training=False)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx, status.is_training:False})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

        with self.subTest(idx=5):
            # train=False with fused=False
            sigma_x = layers.norms.batch_norm(x, epsilon=0.001, fused=False)
            batch_norm = tf.keras.layers.BatchNormalization(fused=False)
            tf_x = batch_norm(x, training=False)
            ops.core.run(self.sess, [tf.global_variables_initializer(), tf.local_variables_initializer()])
            _sigma_x, _tf_x = ops.core.run(self.sess, [sigma_x, tf_x], {x:nx, status.is_training:False})
            self.assertListEqual(_sigma_x.tolist(), _tf_x.tolist())

    @classmethod
    def tearDownClass(cls):
        ops.core.close_session(cls.sess)
