import sys
sys.path.append('/home/xiaox/studio/src/git-series')
import numpy as np
import os
import os.path

from sigma.ops import core

import tensorflow as tf


labelmap = {'airplane': 0,
            'bathtub': 1,
            'bed': 2,
            'bench': 3,
            'bookshelf': 4,
            'bottle': 5,
            'bowl': 6,
            'car': 7,
            'chair': 8,
            'cone': 9,
            'cup': 10,
            'curtain': 11,
            'desk': 12,
            'door': 13,
            'dresser': 14,
            'flower_pot': 15,
            'glass_box':16,
            'guitar':17,
            'keyboard':18,
            'lamp':19,
            'laptop': 20,
            'mantel': 21,
            'monitor':22,
            'night_stand': 23,
            'person': 24,
            'piano': 25,
            'plant': 26,
            'radio': 27,
            'range_hood': 28,
            'sink': 29,
            'sofa': 30,
            'stairs': 31,
            'stool': 32,
            'table': 33,
            'tent': 34,
            'toilet': 35,
            'tv_stand': 36,
            'vase': 37,
            'wardrobe': 38,
            'xbox': 39}

def cutpoints(points, rows, num_points):
    indices = tf.range(rows, dtype=tf.int32)
    indices = tf.random.shuffle(indices)
    points = tf.gather(points, indices)
    slices = np.arange(num_points, dtype=np.int32)
    points = tf.gather(points, slices)
    def _cut():
        return points
    return _cut

def parse(num_points):
    def _parse(record):
        features = tf.parse_single_example(
                record,
                features={
                    'rows': tf.FixedLenFeature([], tf.int64),
                    'cols': tf.FixedLenFeature([], tf.int64),
                    'points': tf.VarLenFeature(tf.float),
                    'labels': tf.FixedLenFeature([], tf.int64)
                    }
                )
        points = tf.sparse_tensor_to_dense(features['points'])
        category = tf.cast(features['label'], tf.int32)
        rows = features['rows']
        cols = features['cols']
        points = tf.reshape(points, [rows, cols])
        points = tf.cond(tf.greater(rows, num_points), cutpoints(points, rows, num_points), lambda :points)
        return points, tf.one_hot(category, 40)
    return _parse

def load_vertices(filename):
    print(filename)
    with open(filename) as fp:
        line = fp.readline().rstrip()
        skiprow = 2
        if len(line) > 3:
            line = line[3:]
            skiprow = 1
        else:
            line = fp.readline()
        line = line.split(' ')[0].lstrip().rstrip()
        rows = int(line)
    return np.loadtxt(filename, dtype=np.float32, skiprows=skiprow, max_rows=rows)

def load(filename, npoints=2048, normalize=True):
    category = filename.split('/')[-3]
    label = labelmap[category]
    points = load_vertices(filename)
    if normalize:
        centroid = points.mean(axis=0)
        points -= centroid
        points /= np.max(np.sqrt(np.sum(points**2, axis=1)))
    size = len(points)
    if size < npoints:
        points = np.concatenate((points, np.zeros((npoints-size, 3))), axis=0)
    return points, label


def create_records(fold, filename, dataset='train', normalize=True):
    writer = core.feature_writer(filename)
    for root, dirs, files in os.walk(fold):
        if root.split('/')[-1] == dataset:
            for f in files:
                points, label = load(os.path.join(root, f), normalize)
                example = core.make_feature({
                    'rows': core.int64_feature([len(points)]),
                    'cols': core.int64_feature([3]),
                    'points': core.float_feature(points.flatten()),
                    'label': core.int64_feature([label])
                    })
                writer.write(example)
    writer.close()

if __name__ == '__main__':
    create_records('/home/xiaox/studio/db/modelnet/40', '/home/xiaox/studio/db/modelnet/train_normalize.tfrecotd')
    create_records('/home/xiaox/studio/db/modelnet/40', '/home/xiaox/studio/db/modelnet/train.tfrecotd', normalize=False)
    create_records('/home/xiaox/studio/db/modelnet/40', '/home/xiaox/studio/db/modelnet/test_normalize.tfrecotd',dataset='test')
    create_records('/home/xiaox/studio/db/modelnet/40', '/home/xiaox/studio/db/modelnet/test.tfrecotd', normalize=False, dataset='test')
