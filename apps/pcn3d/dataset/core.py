import os.path
import tensorflow as tf

def get_tfrecord_size(filename):
    count = 0
    for _ in tf.python_io.tf_record_iterator(filename):
        count += 1
    return count

def prepare_dataset(num_points, batch_size, epochs, setname='shapenet_part'):
    base = '/home/xiaox/studio/db'
    if setname == 'shapenet_part':
        from .shapenet_part import parse
        train_filename = os.path.join(base, 'shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized.tfrecord')
        valid_filename = os.path.join(base, 'shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_valid.tfrecord')
        tests_filename = os.path.join(base, 'shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_test.tfrecord')
        valid_iters = get_tfrecord_size(valid_filename)
    elif setname == 'modelnet40':
        from .modelnet40_loader import parse
        train_filename = os.path.join(base, 'modelnet/train2048.tfrecord')
        valid_filename = None
        tests_filename = os.path.join(base, 'modelnet/test2048.tfrecord')
        valid_iters = None
    else:
        raise ValueError('dataset `{}` not support'.format(setname))
    train_iters = get_tfrecord_size(train_filename)
    tests_iters = get_tfrecord_size(tests_filename)
    filename = tf.placeholder(tf.string, shape=[])
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse(num_points))
    dataset = dataset.shuffle(1000).batch(batch_size)
    dataset = dataset.repeat(epochs)
    iterator = dataset.make_initializable_iterator()

    return train_filename, valid_filename, tests_filename,\
           train_iters, valid_iters, tests_iters, \
           filename, iterator
