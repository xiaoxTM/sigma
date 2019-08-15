import tensorflow as tf
import numpy as np

nparray = np.random.randn(20, 28, 28, 5)
tfarray = tf.placeholder(dtype=tf.float64, shape=(20, 28, 28, 5))
tfslice = tf.gather(tfarray, 2, axis=-1)
npslice = nparray[:, :, :, 2]

with tf.Session() as sess:
    tfslice_ = sess.run(tfslice, feed_dict={tfarray: nparray})
    for i in range(20):
        print(npslice[i]-tfslice_[i])

