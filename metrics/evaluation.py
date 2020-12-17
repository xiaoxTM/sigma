import numpy as np
import logging

try:
    from sklearn import metrics
    from sklearn.metrics import pairwise

    def confusion_matrix(preds, trues, *args, **kwargs):
        return metrics.confusion_matrix(trues, preds, *args, **kwargs)

    def accuracy_score(preds, trues, *args, **kwargs):
        return metrics.accuracy_score(trues, preds, *args, **kwargs)

    def mean_accuracy_score(preds, trues, *args, **kwargs):
        return metrics.balanced_accuracy_score(trues, preds, *args, **kwargs)

    def cosine_similarity(x, y=None, *args, **kwargs):
        return pairwise.cosine_similarity(x, y, *args, **kwargs)

    def euclidean_distances(x, y=None, *args, **kwargs):
        return pairwise.euclidean_distances(x, y, *args, **kwargs)

except Exception as e:
    logging.warning('scikit-learn module not installed')


def topk(x, k, axis=1):
    assert axis in [0, 1]
    size = x.shape[axis]
    assert k > 0 and k <= size
    part = np.argpartition(x, k, axis=axis)
    if axis==0:
        index = np.arange(x.shape[1-axis])
        argsort_k = np.argsort(x[part[0:k, :], index], axis=axis)
        return part[0:k, :][argsort_k, index]
    else:
        index = np.arange(x.shape[1-axis])
        argsort_k = np.argsort(x[index, part[:, 0:k]], axis=axis)
        return part[:, 0:k][index, argsort_k]


def hitmap(predict, label, nclass):
    if not isinstance(predict, np.ndarray):
        raise TypeError('`predict` must be np.ndarray, given {}'
                        .format(type(predict)))
    if not isinstance(label, np.ndarray):
        raise TypeError('`label` must be np.ndarray, given {}'
                        .format(type(label)))
    if len(predict.shape) == 2:
        predict = np.argmax(predict, axis=1)
    elif len(predict.shape) > 2:
        raise ValueError('`predict` shape must be 1/2, given {}'.format(predict.shape))
    if len(label.shape) == 2:
        label = np.argmax(label, axis=1)
    elif len(label.shape) > 2:
        raise ValueError('`label` shape must be 1/2, given {}'.format(label.shape))

    # predict / label: [batch-size]
    matrix = np.zeros(shape=(nclass, nclass))
    for p, l in zip(predict, label):
        matrix[p, l] += 1
    return matrix

def accuracy(predict, label):
    if not isinstance(predict, np.ndarray):
        raise TypeError('`predict` must be np.ndarray, given {}'
                        .format(type(predict)))
    if not isinstance(label, np.ndarray):
        raise TypeError('`label` must be np.ndarray, given {}'
                        .format(type(label)))
    if len(predict.shape) == 2:
        predict = np.argmax(predict, axis=1)
    elif len(predict.shape) > 2:
        raise ValueError('`predict` shape must be 1/2, given {}'.format(predict.shape))
    if len(label.shape) == 2:
        label = np.argmax(label, axis=1)
    elif len(label.shape) > 2:
        raise ValueError('`label` shape must be 1/2, given {}'.format(label.shape))

    predict = predict.flatten()
    label = label.flatten()
    correct = np.sum(np.where(predict == label, np.ones_like(predict), np.zeros_like(predict)))
    return  correct * 1.0 / len(predict)


def mean_perclass_accuracy(apc):
    if not isinstance(apc, (np.ndarray, list, tuple)):
        raise TypeError('`apc` must be np.ndarray / list /tuple, given {}'
                        .format(type(apc)))
    if isinstance(apc, (list, tuple)):
        apc = np.asarray(apc)
    if len(apc.shape) != 1:
        raise ValueError('`apc` dimension must be 1, given {}'.format(len(apc)))
    nan_idx = np.argwhere(np.isnan(apc))
    apc[nan_idx] = 0
    return np.sum(apc) / (len(apc) - len(nan_idx))


def accuracy_from_hitmap(hitmap, label='row', mean=False):
    if not isinstance(hitmap, np.ndarray):
        raise TypeError('`hitmap` must be np.ndarray, given {}'
                        .format(type(hitmap)))
    if len(hitmap.shape) != 2:
        raise ValueError('`hitmap` dimension must be 2, given {}'.format(len(hitmap)))
    if hitmap.shape[0] != hitmap.shape[1]:
        raise ValueError('`hitmap` row must euqal col, given {}'.format(hitmap.shape))
    if label not in ['row', 'col']:
        raise ValueError('`label` must be `row` or `col`, given {}'.format(label))
    diag = np.diag(hitmap)
    correct = np.sum(diag)
    total = np.sum(hitmap)
    acc = correct * 1.0 / total
    if label == 'row':
        class_correct = np.sum(hitmap, axis=0)
    else:
        class_correct = np.sum(hitmap, axis=1)
    # perclass accuracy
    pca = diag / class_correct
    if mean:
        return acc, pca, mean_perclass_accuracy(pca)
    return acc, pca
