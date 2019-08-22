import sys
sys.path.append('/home/xiaox/studio/src/git-series')

from sigma.ops import core
import tensorflow as tf
from modelnet40_loader import ModelNetH5Dataset as modelnet40

def create_tfrecord(filename, root, train=True):
    dataset = modelnet40(root, train=train, batch_size=1, npoints=2048)
    writer = core.feature_writer(filename)
    while dataset.has_next_batch():
        points, label = dataset.next_batch()
        example = core.make_feature({
            'points': core.float_feature(points.flatten()),
            'label': core.int64_feature([label])
            })
        writer.write(example)
    writer.close()

def parse(num_points=2048, onehot=True):
    def _parse(record):
        features = tf.parse_single_example(
                record,
                features={
                    'points': tf.VarLenFeature(tf.float32),
                    'label': tf.FixedLenFeature([], tf.int64)
                    }
                )
        points = tf.sparse_tensor_to_dense(features['points'])
        points = core.reshape(points, [num_points, 3])
        label = tf.cast(features['label'], tf.int32)
        if onehot:
            label = tf.one_hot(label, 40)
        return points, label
    return _parse

if __name__ == '__main__':
    if True:
        create_tfrecord('/mnt/nas/xiaox/db/modelnet/train2048.tfrecord',
                '/home/xiaox/studio/db/modelnet/ply_hdf5_2018/', train=True)
        create_tfrecord('/mnt/nas/xiaox/db/modelnet/test2048.tfrecord',
                '/home/xiaox/studio/db/modelnet/ply_hdf5_2018/', train=False)
    else:
        filename = tf.placeholder(tf.string, shape=[])
        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(parse())
        dataset = dataset.shuffle(1000).batch(20)
        iterator = dataset.make_initializable_iterator()
        inputs, labels = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer, feed_dict={filename:'/mnt/nas/xiaox/db/modelnet/train2048.tfrecord'})
            points, label = sess.run([inputs, labels])
            print(points.shape, label.shape)
