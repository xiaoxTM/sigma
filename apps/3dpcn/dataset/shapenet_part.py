import sys
sys.path.append('/home/xiaox/studio/src/git-series')
from sigma.ops import core

import numpy as np
import multiprocessing as mp
import os.path
import json

global_root = None
fold2id = None
npoints=None
normalization=None

def one_hot(x, nclass):
    y = np.zeros(nclass, dtype=np.int64)
    y[x] = 1
    return y


def _load(root, filename, fold2id, normalization, npoints=2048):
    print('filename:', filename)
    splits = filename.rsplit('/')
    points = np.loadtxt(os.path.join(root, splits[0], 'points', splits[1]+'.pts'))
    if normalization:
        centroid = points.mean(axis=0)
        points -= centroid
        points /= np.max(np.sqrt(np.sum(points**2, axis=1)))
    parts  = np.loadtxt(os.path.join(root, splits[0], 'points_label', splits[1]+'.seg'), dtype=np.int64)
    size = len(points)
    if size < npoints:
        points = np.concatenate((points, np.zeros((npoints-size, 3))), axis=0)
        parts = np.concatenate((parts, np.zeros((npoints-size,))), axis=0)
    category = fold2id[splits[0]]
    return points, parts, category


def create_records(filename,
                   root,
                   splits='shuffled_train_file_list.json',
                   normalize=True):
    categories = np.loadtxt(os.path.join(root, 'synsetoffset2category.txt'), dtype=str)
    fold2id = {fold:cid for cid, (cat, fold) in enumerate(categories)}
    with open(os.path.join(root, 'train_test_split', splits), 'r') as fsplit:
        # splits: shape_data/fold/file_name
        # filenames = {'fold/name', 'fold/name', ...}
        filenames = np.asarray([split.split('/', 1)[1:][0] for split in json.load(fsplit)])
    size = len(filenames)
    writer = core.feature_writer(filename)
    for idx in range(size):
        points, part_labels, category_label = _load(root, filenames[idx], fold2id, normalize)
        example = core.make_feature({
            'rows':core.int64_feature([len(points)]),
            'cols':core.int64_feature([3]),
            'points':core.bytes_feature([points.tostring()]),
            'part_labels':core.bytes_feature([part_labels.tostring()]),
            'category_label':core.int64_feature([category_label]),
            })
        writer.write(example)
    writer.close()


def load(filename):
    splits = filename.rsplit('/')
    points = np.loadtxt(os.path.join(global_root, splits[0], 'points', splits[1]+'.pts'))
    parts  = np.loadtxt(os.path.join(global_root, splits[0], 'points_label', splits[1]+'.seg'))
    category = one_hot(fold2id[splits[0]], len(fold2id))
    size = len(parts)
    if size < npoints:
        points = np.concatenate((points, np.zeros((npoints-size, 3))), axis=0)
        parts = np.concatenate((parts, np.zeros((npoints-size,))), axis=0)
    elif size > npoints:
        index = np.arange(size)
        np.random.shuffle(index)
        points = points[index[:npoints], :]
        parts = parts[index[:npoints]]
    if normalization:
        centroid = points.mean(axis=0)
        points -= centroid
        points /= np.max(np.sqrt(np.sum(points**2, axis=1)))
    return (points, parts, category)


def dataset(root,
            splits='shuffled_train_file_list.json',
            batchsize=8,
            num_points=2048,
            shuffle=True,
            normalize=True):
    global global_root
    global_root = root
    global fold2id
    # [[class, fold], ...]
    # fold2id = {fold: category_id}
    categories = np.loadtxt(os.path.join(root, 'synsetoffset2category.txt'), dtype=str)
    fold2id = {fold:cid for cid, (cat,fold) in enumerate(categories)}
    global npoints
    npoints = num_points
    global normalization
    normalization = normalize
    with open(os.path.join(root, 'train_test_split', splits), 'r') as fsplit:
        # splits: shape_data/fold/file_name
        # filenames = {'fold/name', 'fold/name', ...}
        filenames = np.asarray([split.split('/', 1)[1:][0] for split in json.load(fsplit)])
    size = len(filenames)
    indices = np.arange(size, dtype=np.int32)
    workers = mp.Pool(processes=min(mp.cpu_count(), batchsize))
    iterations = int(size / batchsize)

    def _generate():
        while True:
            if shuffle:
                np.random.shuffle(indices)
            for iteration in range(iterations):
                beg = iteration * batchsize
                if beg > size:
                    beg = size - batchsize
                end = max(min(beg+batchsize, size), beg)
                names = filenames[indices[beg:end]]
                results = workers.map(load, names.tolist())
                points = []
                parts = []
                categories = []
                for result in results:
                    points.append(result[0])
                    parts.append(result[1])
                    categories.append(result[2])
                yield np.stack(points, axis=0), np.stack(parts), np.asarray(categories)

    return _generate(), iterations


if __name__ == '__main__':
    #dataloader, _ = dataset(root='/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0', batchsize=100)
    #for i in range(100000):
    #    print('i:', i)
    #    points, parts, categories=next(dataloader)
    #    print('points:', points.shape)
    #    print('parts:', parts.shape)
    #    print('categories:', categories.shape)
    create_records('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized.tfrecord',
                   '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0')
    create_records('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_valid.tfrecord',
                   '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                   splits='shuffled_val_file_list.json',
                   )
    create_records('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_normalized_test.tfrecord',
                   '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                   splits='shuffled_test_file_list.json',
                   )
    create_records('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation.tfrecord',
                   '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                   normalize=False)
    create_records('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_valid.tfrecord',
                   '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                   splits='shuffled_val_file_list.json',
                   normalize=False)
    create_records('/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_test.tfrecord',
                   '/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0',
                   splits='shuffled_test_file_list.json',
                   normalize=False)

