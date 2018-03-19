import tensorflow as tf
from . import helper, core


def metric(fun):
    def _metric(from_logits=True,
                weights=None,
                metrics_collections=None,
                updates_collections=None,
                reuse=False,
                name=None,
                scope=None,
                *args):
        ops_scope, _, name = helper.assign_scope(name,
                                                 scope,
                                                 fun.__name__,
                                                 reuse)
        return fun(from_logits,
                   weights,
                   metrics_collections,
                   updates_collections,
                   reuse,
                   name,
                   ops_scope,
                   *args)
    return _metric


@metric
def accuracy(from_logits=True,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             reuse=False,
             name=None,
             scope=None):
    def _accuracy(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            return tf.metrics.accuracy(labels,
                                       x,
                                       weights,
                                       metrics_collections,
                                       updates_collections,
                                       name)
    return _accuracy


@metric
def auc(from_logits=True,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        reuse=False,
        name=None,
        scope=None,
        num_thresholds=200,
        curve='ROC',
        summation_method='trapezoidal'):
    def _auc(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            return tf.metrics,auc(labels,
                                  x,
                                  weights,
                                  num_thresholds,
                                  metrics_collections,
                                  updates_collections,
                                  curve,
                                  name,
                                  summation_method)
    return _auc


@metric
def false_negatives(from_logits=True,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    reuse=False,
                    name=None,
                    scope=None,
                    thresholds=None):
    def _false_negatives(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            if thresholds is not None:
                return tf.metrics.false_negatives_at_threshold(labels,
                                                               x,
                                                               thresholds,
                                                               weights,
                                                               metrics_collections,
                                                               updates_collections,
                                                               name)
            else:
                return tf.metrics.false_negatives(labels,
                                                  x,
                                                  weights,
                                                  metrics_collections,
                                                  updates_collections,
                                                  name)
    return _false_negatives


@metric
def false_positives(from_logits=True,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    reuse=False,
                    name=None,
                    scope=None,
                    thresholds=None):
    def _false_positives(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            if thresholds is not None:
                return tf.metrics.false_positives_at_threshold(labels,
                                                               x,
                                                               thresholds,
                                                               weights,
                                                               metrics_collections,
                                                               updates_collections,
                                                               name)
            else:
                return tf.metrics.false_positives(labels,
                                                  x,
                                                  weights,
                                                  metrics_collections,
                                                  updates_collections,
                                                  name)
    return _false_positives


@metric
def true_negatives(from_logits=True,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   reuse=False,
                   name=None,
                   scope=None,
                   thresholds=None):
    def _true_negatives(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            if thresholds is not None:
                return tf.metrics.true_negatives_at_threshold(labels,
                                                              x,
                                                              thresholds,
                                                              weights,
                                                              metrics_collections,
                                                              updates_collections,
                                                              name)
            else:
                return tf.metrics.true_negatives(labels,
                                                 x,
                                                 weights,
                                                 metrics_collections,
                                                 updates_collections,
                                                 name)
    return _true_negatives


@metric
def true_positives(from_logits=True,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   reuse=False,
                   name=None,
                   scope=None,
                   thresholds=None):
    def _true_positives(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            if thresholds is not None:
                return tf.metrics.true_positives_at_threshold(labels,
                                                              x,
                                                              thresholds,
                                                              weights,
                                                              metrics_collections,
                                                              updates_collections,
                                                              name)
            else:
                return tf.metrics.true_positives(labels,
                                                 x,
                                                 weights,
                                                 metrics_collections,
                                                 updates_collections,
                                                 name)
    return _true_positives


@metric
def mean_iou(from_logits=True,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             reuse=False,
             name=None,
             scope=None,
             nclass=None): # nclass = None is just for normalizing API
    if nclass is None:
        raise TypeError('`nclass` for `mean_iou` can not be None')
    def _mean_iou(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            return tf.metrics.mean_iou(labels,
                                       x,
                                       nclass,
                                       weights,
                                       metrics_collections,
                                       updates_collections,
                                       name)
    return _mean_iou


@metric
def precision(from_logits=True,
              weights=None,
              metrics_collections=None,
              updates_collections=None,
              reuse=False,
              name=None,
              scope=None):
    def _precision(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            return tf.metrics.precision(labels,
                                        x,
                                        weights,
                                        metrics_collections,
                                        updates_collections,
                                        name)
    return _precision


@metric
def recall(from_logits=True,
           weights=None,
           metrics_collections=None,
           updates_collections=None,
           reuse=False,
           name=None,
           scope=None):
    def _recall(x, labels):
        with scope:
            if from_logits:
                x = core.argmax(x, core.axis)
            labels = core.argmax(labels, core.axis)
            return tf.metrics.recall(labels,
                                     x,
                                     weights,
                                     metrics_collections,
                                     updates_collections,
                                     name)
    return _recall


def get(m, **kwargs):
    """ get loss from None | string | callable function
    """
    if m is None:
        return None
    elif isinstance(m, str):
        return eval('{}(**kwargs)'.format(m))
    elif helper.is_tensor(m) or callable(m):
        return m
    elif isinstance(m, (list, tuple)):
        # tf.metrics.* that includes
        # metric, update_op
        # as tuple
        return m
    else:
        raise ValueError('cannot get metric `{}` with type {}'
                         .format(m, type(m)))
