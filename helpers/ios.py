import tensorflow as tf
import os.path
from .. import colors, layers, status
from ..ops import helper
import numpy as np
import sigma
import h5py
import pickle
import gzip
import io


def encode(strings, codec='utf8'):
    if isinstance(strings, str):
        return strings.encode(codec)
    elif isinstance(strings, (list, tuple, np.ndarray)):
        def _encode(string):
            if isinstance(string, str):
                return string.encode(codec)
            elif isinstance(string, np.bytes_):
                return string.tostring().encode(codec)
            else:
                raise TypeError('`string` must be string, given {}'
                                .format(type(string)))
        return list(map(_encode, strings))
    else:
        raise TypeError('`strings` must be string or '
                        'list/tuple of string, given {}'
                        .format(type(strings)))


def decode(strings, codec='utf8'):
    if isinstance(strings, str):
        return strings.decode(codec)
    elif isinstance(strings, (list, tuple, np.ndarray)):
        def _decode(string):
            if isinstance(string, str):
                return string.decode(codec)
            elif isinstance(string, np.bytes_):
                return string.tostring().decode(codec)
            else:
                raise TypeError('`string` must be string, given {}'
                                .format(type(string)))
        return list(map(_decode, strings))
    else:
        raise TypeError('`strings` must be string or '
                        'list/tuple of string, given {}'
                        .format(type(strings)))


def load(session, checkpoints,
         saver=None,
         verbose=True):
    if saver is None:
        saver = tf.train.Saver()
    if not isinstance(saver, tf.train.Saver):
        raise TypeError('`{}saver{}` must be instance of {}tf.train.Saver{}. given {}{}{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.fg.red, type(saver), colors.reset)
                        )
    if not isinstance(session, tf.Session):
        raise TypeError('`{}session{}` must be instance of {}tf.Session{}. given {}{}{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.fg.red, type(session), colors.reset)
                        )
    if not os.path.isdir(checkpoints):
        raise FileNotFoundError('Directory {}{}{} not found'
                                .format(colors.fg.red,
                                        checkpoints,
                                        colors.reset))
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoints))
    if ckpt and ckpt.model_checkpoint_path:
        if verbose:
            print('{}load check point from {}{}{}'
                   .format(colors.fg.cyan, colors.fg.red,
                           ckpt.model_checkpoint_path, colors.reset)
                 )
        saver.restore(session, ckpt.model_checkpoint_path)
    elif verbose:
        print('{}restoring from checkpoint ignored{}'
              .format(colors.fg.red, colors.reset))
    return session, saver


def save(session, checkpoints,
         saver=None,
         verbose=True,
         **kwargs):
    if saver is None:
        saver = tf.train.Saver()
    if not isinstance(saver, tf.train.Saver):
        raise TypeError('`{}saver{}` must be instance of {}tf.train.Saver{}. given {}{}{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.fg.red, type(saver), colors.reset)
                        )
    if not isinstance(session, tf.Session):
        raise TypeError('`{}session{}` must be instance of {}tf.Session{}. given {}{}{}'
                        .format(colors.fg.green, colors.reset,
                                colors.fg.blue, colors.reset,
                                colors.fg.red, type(session), colors.reset)
                        )
    if not os.path.isdir(checkpoints):
        raise FileNotFoundError('Directory {}{}{} not found'
                                .format(colors.fg.red, checkpoints, colors.reset))
    if verbose:
        print('{}saving check point to {}{}{}'
               .format(colors.fg.cyan, colors.fg.red,
                       checkpoints, colors.reset)
             )
    saver.save(session, checkpoints, **kwargs)
    return session, saver


def import_weights(filename, session,
                   graph=None,
                   collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                   verbose=True):
    if graph is None:
        graph = session.graph
    if collections is None:
        collections = graph.get_all_collection_keys()
    elif isinstance(collections, str):
        collections = [collections]

    if not isinstance(collections, (list, tuple, np.ndarray)):
        raise TypeError('collections must be list/tuple. given {}{}{}'
                        .format(colors.fg.red, type(collections), colors.reset))
    with h5py.File(filename, 'r') as f:
        if verbose:
            print('importing weights from {}{}{}'
                  .format(colors.fg.green, filename, colors.reset))
        imported_weights = {}
        for collection in collections:
            weight_group = f[collection]
            weight_names = decode(weight_group.attrs['weight_names'])
            params = graph.get_collection(collection)
            for param in params:
                if not imported_weights.get(param.name, False):
                    if param.name in weight_names:
                        value = np.asarray(weight_group[param.name])
                        op = tf.assign(param, value)
                        session.run(op)
                    elif verbose:
                        print('parameter {}{}{} not found.'
                              .format(colors.fg.red, param.name, colors.reset))
                    imported_weights[param.name] = True
    return graph, session


def export_weights(filename, session,
                   graph=None,
                   collections=[tf.GraphKeys.GLOBAL_VARIABLES],
                   verbose=True):
    with h5py.File(filename, mode='w') as f:
        if verbose:
            print('exporting weights to {}{}{}'
                  .format(colors.fg.green, filename, colors.reset))
        if graph is None:
            graph = session.graph
        f.attrs['sigma_version'] = sigma.__version__.encode('utf-8')
        f.attrs['data_format'] = status.data_format.encode('utf-8')
        if collections is None:
            collections = graph.get_all_collection_keys()
        elif isinstance(collections, str):
            collections = [collections]
        if not isinstance(collections, (list, tuple, np.ndarray)):
            raise TypeError('`collections` must be list/tuple or np.ndarray')
        exported_weights = {}
        for collection in collections:
            weight_group = f.create_group(collection)
            params = graph.get_collection(collection)
            names = []
            for param in params:
                if not exported_weights.get(param.name, False):
                    # print('param', param)
                    val = session.run(param)
                    # print('val:', val)
                    pset = weight_group.create_dataset(str(param.name), val.shape,
                                                       dtype=val.dtype)
                    names.append(param.name)
                    if not val.shape:
                        pset[()] = val
                    else:
                        pset[:] = val
                    exported_weights[param.name] = True
            if len(names) != 0:
                weight_group.attrs['weight_names'] = encode(names)


def import_model(filename, session,
                 verbose=True,
                 **kwargs):
    if verbose:
        print('importing model from {}{}{}'
              .format(colors.fg.green, filename, colors.reset))
    pkl = gzip.open(filename, 'rb')
    meta, data = pickle.load(pkl, **kwargs)
    pkl.close()
    with io.StringIO(meta) as sio:
        saver = tf.train.import_meta_graph(sio, **kwargs)
    with io.StringIO(data) as sio:
        saver.restore(session, sio)


def export_model(filename, session,
                 verbose=True,
                 **kwargs):
    if verbose:
        print('exporting model to {}{}{}'
              .format(colors.fg.green, filename, colors.reset))
    if saver is None:
        saver = tf.train.Saver()
    pkl = gzip.open(filename, 'wb')
    meta = tf.train.export_meta_graph(**kwargs)
    with io.StringIO() as sio:
        saver.save(session, sio)
        data = sio.getvalue()
        pkl.dummy([meta, data])
    pkl.close()


def export_graph(filename, ext=None):
    helper.export_graph(filename, ext)
