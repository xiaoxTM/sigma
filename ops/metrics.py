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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            x = core.metrics_accuracy(labels,
                                      x,
                                      weights,
                                      metrics_collections,
                                      updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
            #return *x
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            x = core.metrics_auc(labels,
                                 x,
                                 weights,
                                 num_thresholds,
                                 metrics_collections,
                                 updates_collections,
                                 curve,
                                 None,
                                 summation_method)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            if thresholds is not None:
                x = core.metrics_false_negatives_at_threshold(labels,
                                                              x,
                                                              thresholds,
                                                              weights,
                                                              metrics_collections,
                                                              updates_collections)
            else:
                x = core.metrics_false_negatives(labels,
                                                 x,
                                                 weights,
                                                 metrics_collections,
                                                 updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            if thresholds is not None:
                x = core.metrics_false_positives_at_threshold(labels,
                                                              x,
                                                              thresholds,
                                                              weights,
                                                              metrics_collections,
                                                              updates_collections)
            else:
                x = core.metrics_false_positives(labels,
                                                 x,
                                                 weights,
                                                 metrics_collections,
                                                 updates_collections)
            variables = core.get_collection(Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            if thresholds is not None:
                x = core.metrics_true_negatives_at_threshold(labels,
                                                             x,
                                                             thresholds,
                                                             weights,
                                                             metrics_collections,
                                                             updates_collections)
            else:
                x = core.metrics_true_negatives(labels,
                                                x,
                                                weights,
                                                metrics_collections,
                                                updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            if thresholds is not None:
                x = core.metrics_true_positives_at_threshold(labels,
                                                             x,
                                                             thresholds,
                                                             weights,
                                                             metrics_collections,
                                                             updates_collections)
            else:
                x = core.metrics_true_positives(labels,
                                                x,
                                                weights,
                                                metrics_collections,
                                                updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            x = core.metrics_mean_iou(labels,
                                      x,
                                      nclass,
                                      weights,
                                      metrics_collections,
                                      updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            x = core.metrics_precision(labels,
                                       x,
                                       weights,
                                       metrics_collections,
                                       updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
                x = core.argmax(x, core.caxis)
                labels = core.argmax(labels, core.caxis)
            x = core.metrics_recall(labels,
                                    x,
                                    weights,
                                    metrics_collections,
                                    updates_collections)
            variables = core.get_collection(core.Collections.local_variables, name)
            initializer = core.variables_initializer(var_list=variables)
            return (*x, initializer)
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
        # metrics.* that includes
        # metric, update_op
        # as tuple
        return m
    else:
        raise ValueError('cannot get metric `{}` with type {}'
                         .format(m, type(m)))
