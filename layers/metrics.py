from .. import ops
from . import core


@core.layer
def accuracy(inputs,
             from_logits=True,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             reuse=False,
             name=None,
             scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.accuracy(from_logits,
                                weights,
                                metrics_collections,
                                updates_collections,
                                reuse,
                                name,
                                scope)(inputs, labels)


@core.layer
def auc(inputs,
        from_logits=True,
        weights=None,
        num_thresholds=200,
        metrics_collections=None,
        updates_collections=None,
        curve='ROC',
        summation_method='trapezoidal',
        reuse=False,
        name=None,
        scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.auc(from_logits,
                           weights,
                           metrics_collections,
                           updates_collections,
                           reuse,
                           name,
                           scope,
                           num_thresholds,
                           curve,
                           summation_method)(inputs, labels)


@core.layer
def false_negatives(inputs,
                    from_logits=True,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    reuse=False,
                    name=None,
                    scope=None,
                    thresholds=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.false_negatives(from_logits,
                                       weights,
                                       metrics_collections,
                                       updates_collections,
                                       reuse,
                                       name,
                                       scope,
                                       thresholds)(inputs, labels)


@core.layer
def false_positives(inputs,
                    from_logits=True,
                    weights=None,
                    metrics_collections=None,
                    updates_collections=None,
                    reuse=False,
                    name=None,
                    scope=None,
                    thresholds=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.false_positives(from_logits,
                                       weights,
                                       metrics_collections,
                                       updates_collections,
                                       reuse,
                                       name,
                                       scope,
                                       thresholds)(inputs, labels)


@core.layer
def true_negatives(inputs,
                   from_logits=True,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   reuse=False,
                   name=None,
                   scope=None,
                   thresholds=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.true_negatives(from_logits,
                                      weights,
                                      metrics_collections,
                                      updates_collections,
                                      reuse,
                                      name,
                                      scope,
                                      thresholds)(inputs, labels)


@core.layer
def true_positives(inputs,
                   from_logits=True,
                   weights=None,
                   metrics_collections=None,
                   updates_collections=None,
                   reuse=False,
                   name=None,
                   scope=None,
                   thresholds=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.true_positives(from_logits,
                                      weights,
                                      metrics_collections,
                                      updates_collections,
                                      reuse,
                                      name,
                                      scope,
                                      thresholds)(inputs, labels)


@core.layer
def mean_iou(inputs,
             from_logits=True,
             weights=None,
             metrics_collections=None,
             updates_collections=None,
             reuse=False,
             name=None,
             scope=None,
             nclass=None): # nclass = None is just for normalizing API
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.mean_iou(from_logits,
                                weights,
                                metrics_collections,
                                updates_collections,
                                reuse,
                                name,
                                scope,
                                nclass)(inputs, labels)


@core.layer
def precision(inputs,
              from_logits=True,
              weights=None,
              metrics_collections=None,
              updates_collections=None,
              reuse=False,
              name=None,
              scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.precision(from_logits,
                                 weights,
                                 metrics_collections,
                                 updates_collections,
                                 reuse,
                                 name,
                                 scope)(inputs, labels)

@core.layer
def recall(inputs,
           from_logits=True,
           weights=None,
           metrics_collections=None,
           updates_collections=None,
           reuse=False,
           name=None,
           scope=None):
    inputs, labels = core.split_inputs(inputs)
    return ops.metrics.recall(from_logits,
                              weights,
                              metrics_collections,
                              updates_collections,
                              reuse,
                              name,
                              scope)(inputs, labels)


def get(m, inputs, labels, **kwargs):
    """ get loss from None | string | callable function
    """
    if m is None:
        return None
    elif isinstance(m, str):
        return eval('{}([inputs, labels], **kwargs)'.format(m))
    elif core.helper.is_tensor(m) or callable(m):
        return m
    else:
        raise ValueError('cannot get metrics `{}` with type {}'
                         .format(m, type(m)))
