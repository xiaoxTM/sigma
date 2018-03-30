import pickle
import os.path
import numpy as np


def load(dirs, to_tensor=True, onehot=False, nclass=None, coarse=True):
    data = []
    def _load(filename):
        with open(os.path.join(dirs, filename), 'rb') as fp:
            datum = pickle.load(fp, encoding='bytes')
            rawdata = datum[b'data'].reshape([-1, 3, 32, 32])
            rawdata = np.transpose(rawdata, (0, 2, 3, 1))
            if coarse:
                labels = datum[b'coarse_labels']
            else:
                labels = datum[b'fine_labels']
            if onehot:
                depth = nclass
                if depth is None:
                    depth = len(np.unique(labels))
                labels = helpers.one_hot(labels, depth)
            if to_tensor:
                rawdata = ops.core.to_tensor(rawdata, ops.core.float32)
        return rawdata, np.asarray(labels)
    xtrain, ytrain = _load('train')
    xvalid, yvalid = _load('test')
    return [xtrain, ytrain], [xvalid, yvalid]
