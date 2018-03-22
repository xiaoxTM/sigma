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

#sepal_length,sepal_width,petal_length,petal_width,species
import numpy as np

from sigma import helpers

setosa = 0
versicolor = 1
virginica = 2

id2name = {setosa: "setosa",
           versicolor: "versicolor",
           virginica: "virgincia"
          }

name2id = {"setosa": setosa,
           "versicolor": versicolor,
           "virginica": virginica
          }

def load(onehot=True):
    samples = np.asarray([
        [5.1,3.5,1.4,0.2], [4.9,3.0,1.4,0.2], [4.7,3.2,1.3,0.2],
        [4.6,3.1,1.5,0.2], [5.0,3.6,1.4,0.2], [5.4,3.9,1.7,0.4],
        [4.6,3.4,1.4,0.3], [5.0,3.4,1.5,0.2], [4.4,2.9,1.4,0.2],
        [4.9,3.1,1.5,0.1], [5.4,3.7,1.5,0.2], [4.8,3.4,1.6,0.2],
        [4.8,3.0,1.4,0.1], [4.3,3.0,1.1,0.1], [5.8,4.0,1.2,0.2],
        [5.7,4.4,1.5,0.4], [5.4,3.9,1.3,0.4], [5.1,3.5,1.4,0.3],
        [5.7,3.8,1.7,0.3], [5.1,3.8,1.5,0.3], [5.4,3.4,1.7,0.2],
        [5.1,3.7,1.5,0.4], [4.6,3.6,1.0,0.2], [5.1,3.3,1.7,0.5],
        [4.8,3.4,1.9,0.2], [5.0,3.0,1.6,0.2], [5.0,3.4,1.6,0.4],
        [5.2,3.5,1.5,0.2], [5.2,3.4,1.4,0.2], [4.7,3.2,1.6,0.2],
        [4.8,3.1,1.6,0.2], [5.4,3.4,1.5,0.4], [5.2,4.1,1.5,0.1],
        [5.5,4.2,1.4,0.2], [4.9,3.1,1.5,0.1], [5.0,3.2,1.2,0.2],
        [5.5,3.5,1.3,0.2], [4.9,3.1,1.5,0.1], [4.4,3.0,1.3,0.2],
        [5.1,3.4,1.5,0.2], [5.0,3.5,1.3,0.3], [4.5,2.3,1.3,0.3],
        [4.4,3.2,1.3,0.2], [5.0,3.5,1.6,0.6], [5.1,3.8,1.9,0.4],
        [4.8,3.0,1.4,0.3], [5.1,3.8,1.6,0.2], [4.6,3.2,1.4,0.2],
        [5.3,3.7,1.5,0.2], [5.0,3.3,1.4,0.2], [7.0,3.2,4.7,1.4],
        [6.4,3.2,4.5,1.5], [6.9,3.1,4.9,1.5], [5.5,2.3,4.0,1.3],
        [6.5,2.8,4.6,1.5], [5.7,2.8,4.5,1.3], [6.3,3.3,4.7,1.6],
        [4.9,2.4,3.3,1.0], [6.6,2.9,4.6,1.3], [5.2,2.7,3.9,1.4],
        [5.0,2.0,3.5,1.0], [5.9,3.0,4.2,1.5], [6.0,2.2,4.0,1.0],
        [6.1,2.9,4.7,1.4], [5.6,2.9,3.6,1.3], [6.7,3.1,4.4,1.4],
        [5.6,3.0,4.5,1.5], [5.8,2.7,4.1,1.0], [6.2,2.2,4.5,1.5],
        [5.6,2.5,3.9,1.1], [5.9,3.2,4.8,1.8], [6.1,2.8,4.0,1.3],
        [6.3,2.5,4.9,1.5], [6.1,2.8,4.7,1.2], [6.4,2.9,4.3,1.3],
        [6.6,3.0,4.4,1.4], [6.8,2.8,4.8,1.4], [6.7,3.0,5.0,1.7],
        [6.0,2.9,4.5,1.5], [5.7,2.6,3.5,1.0], [5.5,2.4,3.8,1.1],
        [5.5,2.4,3.7,1.0], [5.8,2.7,3.9,1.2], [6.0,2.7,5.1,1.6],
        [5.4,3.0,4.5,1.5], [6.0,3.4,4.5,1.6], [6.7,3.1,4.7,1.5],
        [6.3,2.3,4.4,1.3], [5.6,3.0,4.1,1.3], [5.5,2.5,4.0,1.3],
        [5.5,2.6,4.4,1.2], [6.1,3.0,4.6,1.4], [5.8,2.6,4.0,1.2],
        [5.0,2.3,3.3,1.0], [5.6,2.7,4.2,1.3], [5.7,3.0,4.2,1.2],
        [5.7,2.9,4.2,1.3], [6.2,2.9,4.3,1.3], [5.1,2.5,3.0,1.1],
        [5.7,2.8,4.1,1.3], [6.3,3.3,6.0,2.5], [5.8,2.7,5.1,1.9],
        [7.1,3.0,5.9,2.1], [6.3,2.9,5.6,1.8], [6.5,3.0,5.8,2.2],
        [7.6,3.0,6.6,2.1], [4.9,2.5,4.5,1.7], [7.3,2.9,6.3,1.8],
        [6.7,2.5,5.8,1.8], [7.2,3.6,6.1,2.5], [6.5,3.2,5.1,2.0],
        [6.4,2.7,5.3,1.9], [6.8,3.0,5.5,2.1], [5.7,2.5,5.0,2.0],
        [5.8,2.8,5.1,2.4], [6.4,3.2,5.3,2.3], [6.5,3.0,5.5,1.8],
        [7.7,3.8,6.7,2.2], [7.7,2.6,6.9,2.3], [6.0,2.2,5.0,1.5],
        [6.9,3.2,5.7,2.3], [5.6,2.8,4.9,2.0], [7.7,2.8,6.7,2.0],
        [6.3,2.7,4.9,1.8], [6.7,3.3,5.7,2.1], [7.2,3.2,6.0,1.8],
        [6.2,2.8,4.8,1.8], [6.1,3.0,4.9,1.8], [6.4,2.8,5.6,2.1],
        [7.2,3.0,5.8,1.6], [7.4,2.8,6.1,1.9], [7.9,3.8,6.4,2.0],
        [6.4,2.8,5.6,2.2], [6.3,2.8,5.1,1.5], [6.1,2.6,5.6,1.4],
        [7.7,3.0,6.1,2.3], [6.3,3.4,5.6,2.4], [6.4,3.1,5.5,1.8],
        [6.0,3.0,4.8,1.8], [6.9,3.1,5.4,2.1], [6.7,3.1,5.6,2.4],
        [6.9,3.1,5.1,2.3], [5.8,2.7,5.1,1.9], [6.8,3.2,5.9,2.3],
        [6.7,3.3,5.7,2.5], [6.7,3.0,5.2,2.3], [6.3,2.5,5.0,1.9],
        [6.5,3.0,5.2,2.0], [6.2,3.4,5.4,2.3], [5.9,3.0,5.1,1.8]
    ])

    labels = np.asarray([
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, setosa, setosa, setosa, setosa,
        setosa, setosa, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, versicolor, versicolor,
        versicolor, versicolor, versicolor, versicolor, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica,
        virginica, virginica, virginica, virginica, virginica, virginica
    ])

    if onehot:
        labels = helpers.one_hot(labels, len(id2name))

    return samples, labels

def generator(batch_size, shuffle=True, onehot=True):
    samples, labels = load(onehot)
    datasize = len(samples.shape[0])
    epochs = int(batch_size / datasize)
    while True:
        index = np.arange(datasize)
        if shuffle:
            np.random.shuffle(index)
        for epoch in np.arange(0, datasize, epochs):
            idx = index[epoch: epoch+batch_size]
            sample = samples[idx, :]
            label = labels[idx, :]
            yield sample, label
