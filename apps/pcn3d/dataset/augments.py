import tensorflow as tf
import numpy as np

def rotate_point_cloud(cloud, batch_size):
    ones = tf.ones(shape=(batch_size, ))
    zeros = tf.zeros(shape=(batch_size, ))
    rotation_angles = tf.random_uniform(shape=(batch_size,)) * 2 * np.pi
    sinval = tf.sin(rotation_angles)
    cosval = tf.cos(rotation_angles)
    # rotation_matrix: [batch_size, 9]
    rotation_matrix = tf.stack([cosval, zeros, sinval, zeros, ones, zeros, -sinval, zeros, cosval], axis=1)
    # rotation_matrix: [batch_size, 3, 3]
    rotation_matrix = tf.reshape(rotation_matrix, [batch_size, 3, 3])
    # cloud: [batch_size, npoints, 3]
    return tf.matmul(cloud, rotation_matrix)


def jitter_point_cloud(cloud, batch_size, sigma=0.01, clip=0.05):
    assert clip > 0, 'clip must be positive, given {}'.format(clip)
    _, npoints, channels = cloud.get_shape().as_list()
    jittered = tf.clip_by_value(tf.random_normal(shape=(batch_size, npoints, channels), stddev=sigma), -1*clip, clip)
    return jittered + cloud


if __name__ == '__main__':
    with tf.Session() as sess:
        cloud = tf.placeholder(dtype=tf.float32, shape=(2,2048,3))
        rotated = rotate_point_cloud(cloud)
        jitterd = jitter_point_cloud(cloud)
        test1 = np.loadtxt('test1.pts', dtype=np.float32)[:2048, :]
        test2 = np.loadtxt('test2.pts', dtype=np.float32)[:2048, :]
        points = np.stack([test1, test2])
        print(points.shape)

        r,j = sess.run([rotated, jitterd], feed_dict={cloud:points})
        for i in range(2):
            np.savetxt('{}_origin.txt'.format(i), points[i, :, :])
            np.savetxt('{}_rotate.txt'.format(i), r[i, :, :])
            np.savetxt('{}_jitter.txt'.format(i), j[i, :, :])
