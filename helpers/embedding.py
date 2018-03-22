"""
    sigma, a deep neural network framework.
    Copyright (C) 2018  Renwu Gao

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

try:
    from sklearn import manifold, decomposition, ensemble
    from sklearn import discriminant_analysis, random_projection
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
