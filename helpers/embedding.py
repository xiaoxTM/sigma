try:
    from sklearn import manifold, decomposition, ensemble,
                        discriminant_analysis, random_projection
except:
    raise ImportError('Library sklearn not found. To enable it, '
          'please install scikit-learn.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import offsetbox
import matplotlib.pyplot as plt

def plot_embedding(x, y, title=None, ax=None):
    """ plot embeddings give samples
        Attributes
        ==========
        x : array or array-like
            should be shape of [nsamples, nfeatures]
        y : array or array-like
            should be shape of [nsamples, 1]
            where `1` indicates the label of each sample
        title : string | None
                figure title
        ax : Axes | None
             axes of figure
    """
    x_min, x_max = np.min(x, 0), np.max(x, 0)
    x = (x - x_min) / (x_max - x_min)

    if ax is None:
        ax = plt.subplot(111)

    for i in range(x.shape[0]):
        plt.text(x[i, 0], x[i, 1], str(y[i]),
                 color = plt.cm.Set1(y[i] / i),
                 fontdict={'weight':'bold', 'size':9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     shown_images = np.array([[1., 1.]])
    #     for i in range()

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_tsne(x):
    X = TSNE(x)
