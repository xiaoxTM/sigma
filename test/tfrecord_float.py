import tensorflow as tf
import numpy as np
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

if __name__ == '__main__':
    a = []
    writer = tf.python_io.TFRecordWriter('tfrecord_float.tfrecord')
    for i in range(2):
        data = np.random.randn(5, 5)
        print(data.dtype)
        data = data.astype(np.float32)
        flatten = data.flatten()
        a.append(data)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'data':_float_feature(flatten),
                'rows':_int64_feature([data.shape[0]]),
                'cols':_int64_feature([data.shape[1]])
                }
            ))

        writer.write(example.SerializeToString())
    writer.close()

    filename_queue = tf.train.string_input_producer(['tfrecord_float.tfrecord'])
    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)
    features = tf.parse_single_example(serialized, features={
        #'data':tf.FixedLenFeature([], tf.float32),
        'data':tf.VarLenFeature(tf.float32),
        'rows':tf.FixedLenFeature([], tf.int64),
        'cols':tf.FixedLenFeature([], tf.int64)
        })
    d = tf.sparse_tensor_to_dense(features['data'], default_value=0)
    datum = tf.reshape(d, [features['rows'], features['cols']])

    coord = tf.train.Coordinator()

    with tf.Session() as sess:
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(2):
            d_ = sess.run(datum)
            print(d_-a[i])
