from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
#import matplotlib.pyplot as plt
#from skimage.transform import AffineTransform
#from skimage.transform import warp

def random_affine_matrix(sigma, max_translation):
    rotate = np.eye(2) + sigma * np.random.normal(0, 1, size=[2, 2])
    transf = np.random.uniform(-max_translation, max_translation, size=[2, 1])
    homoge = np.array([0., 0., 1.])
    return np.concatenate([np.concatenate([rotate, transf], axis=1), homoge], axis=0)


def transform_data(data_split, mode, max_translation=5, sigma=0.1, show=False):
    assert mode in ['translation', 'affine'], 'transformation {} not support'.format(mode)

    #if show:
    #    plt.ion()
    #    _, [ax1, ax2] = plt.subplots(1, 2)

    samples = []
    labels  = []
    transformations = []

    for i in np.random.permutation(len(data_split)):
        image = np.reshape(data_split[i], (28,28))
        if mode == 'translation':
            xtrans = random.randint(-max_translation, max_translation)
            ytrans = random.randint(-max_translation, max_translation)
            transformation = np.asarray([xtrans, ytrans])
            itrans = np.roll(np.roll(image, xtrans, axis=0), ytrans, axis=1)

        else:
            transformation = random_affine_matrix(sigma, max_translation)
            itrans = warp(image, AffineTransform(matrix=transformation))

        #if show:
        #    ax1.imshow(image)
        #    ax2.imshow(itrans)
        #    plt.show()
        #    plt.waitforbuttonpress()

        samples.append(np.expand_dims(image, axis=-1))
        transformations.append(transformation)
        labels.append(np.expand_dims(itrans, axis=-1))

    return [(samples, transformations), labels]


def generate_mnist_data(mode, max_translation=5, sigma=0.1):

    def _mnist_data(**kwargs):
        mnist = input_data.read_data_sets('/home/xiaox/studio/db/mnist', one_hot=True)

        mnist = {'train': mnist.train.images,
                 'valid': mnist.validation.images,
                 'test':  mnist.test.images}

        train_extra = transform_data(mnist['train'], mode, max_translation, sigma)
        valid_extra = transform_data(mnist['valid'], mode, max_translation, sigma)
        test_extra = transform_data(mnist['test'], mode, max_translation, sigma)
        input_shape = [None]+list(np.asarray(train_extra[0][0]).shape[1:])
        extra_shape = [None]+list(np.asarray(train_extra[0][1]).shape[1:])
        label_shape = [None]+list(np.asarray(train_extra[1]).shape[1:])
        return [(input_shape, extra_shape), label_shape], [train_extra, valid_extra, test_extra]
    return _mnist_data
