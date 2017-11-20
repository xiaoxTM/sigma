import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.mplot3d import Axes3D
from sklearn import (manifold, decomposition, ensemble,
                     discriminant_analysis, random_projection)

import pickle
import gzip

def load(filename):
    f = gzip.open(filename, 'rb')
    objs = pickle.load(f)
    f.close()
    return objs

def plot_embedding(x, axis, ax=None, title=None, filename=None):
    nsamples = int(x.shape[0] / 3)
    real, fake, blend = x[:nsamples], x[nsamples:2*(nsamples)], x[(2*nsamples):]

    if ax is None:
        _, ax = plt.subplots(1, 2, sharey=True, figsize=(8, 4))

    if axis == 0:
        ax[0].clear()
        ax[1].clear()
        ax[0].grid(True)
        ax[1].grid(True)
        ax[0].set_title('initialization')
        ax[1].set_title('generation')

    for u, v in real:
        ax[axis].scatter(u, v, c='r', marker='+')

    for u, v in fake:
        ax[axis].scatter(u, v, c='g', marker='x')

    for u, v in blend:
        ax[axis].scatter(u, v, c='b', marker='+')

    if filename is not None:
        if axis == 1:
            plt.savefig(filename)
    else:
        plt.show()
    return ax

def plot_data(data, filename, axis, ax=None):
    x = data
    x = np.concatenate(x, axis=0)

    x_flatten = np.reshape(x, (-1, 28 * 28))

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_tsne = tsne.fit_transform(x_flatten)
    # print('embedding')
    return plot_embedding(x_tsne, axis, ax, "t-SNE embedding of the digits", filename)

def plot(filename, axis, ax=None):
    # t-SNE embedding of the digits dataset
    # x: real-data, fake-data, blend-data, blend-labels
    x = load(filename)
    return plot_data(x, '{}-tsne.png'.format(filename), axis, ax)

if __name__ == '__main__':
    ax = None
    for i in range(1, 101):
        ax = plot('exp/pickles/{}/0.pkl'.format(i), ax)
