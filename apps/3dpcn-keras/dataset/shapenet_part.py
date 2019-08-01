import numpy as np
import multiprocessing as mp
import os.path
import json

global_root = None
fold2id = None
npoints=None
normalization=None

def load(filename):
    splits = filename.rsplit('/')
    points = np.loadtxt(os.path.join(global_root, splits[0], 'points', splits[1]+'.pts'))
    parts  = np.loadtxt(os.path.join(global_root, splits[0], 'points_label', splits[1]+'.seg'))
    category = fold2id[splits[0]]
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
    dataloader = dataset(root='/home/xiaox/studio/db/shapenet/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0')
    for i in range(100000):
        print('i:', i)
        points, parts, categories=next(dataloader)
        print('points:', points.shape)
        print('parts:', parts.shape)
        print('categories:', categories.shape)
