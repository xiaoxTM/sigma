from sigma.utils.params import expand_param, split_params
from sigma.fontstyles import colors
from sigma.nn.torch import CheckPointer, initializers, regularizers
import os.path
import torch
from torch import nn
from torch import optim
import collections
import numpy as np
import argparse

visualization=True
try:
    from torchviz import make_dot
except Exception as e:
    visualization=False

from . import losses
from sigma.metrics import metrics as protocols

class BaseModule(nn.Module):
    def __init__(self,
                 *args,
                 loss='cce',
                 metric=None,
                 loss_weights=1.0,
                 running_window_size=0,
                 weight_initializer=None,
                 bias_initializer=None,
                 weight_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(BaseModule, self).__init__()
        self.running_window_size=running_window_size
        self._weight_initializer = initializers.get(weight_initializer)
        self._bias_initializer = initializers.get(bias_initializer)
        self._weight_regularizer = regularizers.get(weight_regularizer)
        self._bias_regularizer = regularizers.get(bias_regularizer)
        self._losses = self.build_losses(loss)
        self._loss_weights = self.build_weights(loss_weights)
        self._metrics = self.build_metrics(metric)
        self._best_metric = 0
        self._loss_dict = collections.OrderedDict()
        self._metric_dict = None
        if self._metrics is not None:
            self._metric_dict=collections.OrderedDict()
        self._preds = []
        self._trues = []
        self._exp = None
        self._weight_lists =None
        self._bias_lists = None
        self._input_device = None

    def initialize_parameters(self, weights, biases):
        assert isinstance(weights,(tuple,list)),'weight must be instance of tuple/list, given: {}'.format(type(weights))
        assert isinstance(biases,(tuple,list)),'bias must be instance of tuple/list, given: {}'.format(type(biases))
        if self._weight_initializer is not None:
            for weight in weights:
                self._weight_initializer(weight)
        if self._bias_initializer is not None:
            for bias in biases:
                self._bias_initializer(bias)

    def build_losses(self, loss):
        if loss is None:
            raise ValueError('loss can not be None')
        elif isinstance(loss, (list, tuple)):
            return [losses.get(l) for l in loss]
        else: # str, may as a string lista
            loss = split_params(loss)
            return [losses.get(l) for l in loss]

    def build_metrics(self, metric):
        if metric is None:
            return None
        elif isinstance(metric, (list, tuple)):
            return [protocols.get(m) for m in metric]
        else: # str, may as a string list
            metric = split_params(metric)
            return [protocols.get(m) for m in metric]

    def build_weights(self, weights):
        if weights is None:
            weights = 1.0
        elif isinstance(weights, str):
            weights = split_params(weights)
            weights = [float(w) for w in weights]
        return np.asarray(expand_param(weights, len(self._losses)))

    @classmethod
    def argument_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--loss',type=str,default='cce(reduction="mean")',help='loss list. for customized loss, assign none then constructe by yourself')
        parser.add_argument('--loss-weights',type=str,default='1.0') # for multivalue, pass as '[0.1;0.3;0.23]'
        parser.add_argument('--metric',type=str,default=None)
        parser.add_argument('--weight-initializer',type=str,default=None)
        parser.add_argument('--bias-initializer',type=str,default=None)
        parser.add_argument('--weight-regularizer',type=str,default=None)
        parser.add_argument('--bias-regularizer',type=str,default=None)
        parser.add_argument('--running-window-size',type=int,default=0)
        return parser

    # may be need overwrite
    def predict(self, x, epoch=None, iteration=None):
        return x.argmax(dim=-1)

    # may be need overwrite
    def represent(self, x):
        # return the representation of the given sample, i.e., the feature representation of it, generally as a vector
        raise NotImplementedError('You need to implement represent function of the basic Module class to get the final feature of your model!!!')

    # may be need overwrite
    def prepare_data(self, data, label):
        if self._input_device is None:
            self._input_device = next(self.parameters()).device
        return data.to(self._input_device), label.to(self._input_device).squeeze()

    def on_batch_begin(self, epoch, batch):
        pass

    def on_batch_end(self, epoch, batch):
        step = self._exp._train_step if self.training else self._exp._valid_step
        return self.on_end('iter', step)

    def on_epoch_begin(self, epoch):
        pass

    def on_epoch_end(self, epoch):
        return self.on_end('epoch', epoch)

    def on_end(self, x, step):
        cost = None
        if len(self._losses) > 1:
            cost = collections.OrderedDict()
            for k,v in self._loss_dict.items():
                v = np.mean(v)
                cost.update({k:v})
        metrics = None
        if self.training:
            if self._metric_dict is not None:
                metrics = collections.OrderedDict()
                for k, v in self._metric_dict.items():
                    v = np.mean(v)
                    metrics.update({k:v})
        else:
            preds, trues = self.prepare_metric()
            metrics= self.metrics(preds, trues)
        return cost, metrics

    def clean(self):
        for k, v in self._loss_dict.items():
            self._loss_dict.update({k:[]})
        if self._metric_dict is not None:
            for k, v in self._metric_dict.items():
                self._metric_dict.update({k:[]})
        self._trues = []
        self._preds = []

    # for different type of model, overwrite this function to store intermediate results
    # may need overwrite
    def store(self, pred, true, is_logits=False):
        self._preds.append(self.numpy(self.predict(pred) if is_logits else pred))
        self._trues.append(self.numpy(true))

    # may need overwrite
    def numpy(self, x):
        if isinstance(x, (list, tuple)):
           x = [self._numpy(e) for e in x]
        else:
            x = self._numpy(x)
        return x

    def _numpy(self, x):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        elif torch.is_tensor(x):
            x = x.numpy()
        return x

    def list_weight_bias(self):
        if self._weight_lists is None:
            self._weight_lists = []
            self._bias_lists = []
            for name, param in self.named_parameters():
                if 'weight' in name:
                    self._weight_lists.append(param)
                elif 'bias' in name:
                    self._bias_lists.append(param)

    def regularize(self):
        self.list_weight_bias()
        wreg, breg = 0, 0
        if self._weight_regularizer is not None:
            wreg = self._weight_regularizer(self._weight_lists)
        if self._bias_regularizer is not None:
            breg = self._bias_regularizer(self._bias_lists)
        return (wreg+breg)

    # may need overwrite
    def losses(self, preds, trues, epoch=None, iteration=None):
        ''' return scalar loss
            preds: torch.Tensor
            trues: torch.Tensor
        '''
        if self._losses is None:
            raise NotImplementedError('You need to implement loss function of the basic Module class')
        losses = collections.OrderedDict()
        for loss in self._losses:
            k, v = list(loss(preds, trues).items())[0]
            losses.update({k:v}) # ordered dict
        return losses

    def loss(self, preds, trues, epoch=None, iteration=None):
        ''' return loss
            preds: torch.Tensor
            trues: torch.Tensor
        '''
        # loss in torch.tensor
        losses = self.losses(preds, trues, epoch, iteration)
        self.loss_update(losses)
        sums = 0
        for l, w in zip(list(losses.values()), self._loss_weights):
            sums += (l * w)
        reg = self.regularize()
        return sums + reg

    def loss_update(self, losses):
        assert isinstance(losses, collections.OrderedDict)
        for k,v in list(losses.items()):
            if k not in self._loss_dict.keys():
                self._loss_dict[k] = []
            self._loss_dict[k].append(self.numpy(v))
            if self.running_window_size > 0 and len(self._loss_dict[k]) > self.running_window_size:
                self._loss_dict[k].pop(0)

    # may need overwrite
    def metrics(self, preds, trues):
        ''' return metrics
            preds: list / np.ndarray
            trues: list / np.ndarray
        '''
        if self._metrics is None:
            return None
        else:
            metrics = self._metrics
            if not isinstance(metrics, (list, tuple)):
                metrics = [metrics]
            metric_dict = collections.OrderedDict()
            for metric in metrics:
                metric_dict.update(metric(preds, trues))
            return metric_dict

    def metric(self, preds, trues):
        ''' return metrics
            preds: torch.Tensor
            trues: torch.Tensor
        '''
        metric_dict = self.metrics(self.numpy(preds), self.numpy(trues))
        if metric_dict is not None:
            self.metric_update(metric_dict)
        return metric_dict

    def metric_update(self, metrics):
        assert isinstance(metrics, collections.OrderedDict)
        for k, v in list(metrics.items()):
            if k not in self._metric_dict.keys():
                self._metric_dict[k] = []
            self._metric_dict[k].append(v)
            if self.running_window_size > 0 and len(self._metric_dict[k]) > self.running_window_size:
                self._metric_dict[k].pop(0)

    def is_better_metric(self):
        if self.training:
            logging.warning('comparison metrics when training is not recommendated')
        if self._metrics is None:
            metric = np.mean(list(self._loss_dict.items())[0][1])
            return metric < self._best_metric
        else:
            preds, trues = self.prepare_metric()
            metric = list(self._metrics[0](preds, trues).items())[0][1]
            return metric > self._best_metric

    def validate_metric(self):
        if self.training:
            logging.warning('validate metrics when training is not recommendated')
        is_better = False
        if self._metrics is None: # use loss as metric
            metric = np.mean(list(self._loss_dict.items())[0][1])
            if metric < self._best_metric:
                self._best_metric = metric
                is_better = True
        else:
            preds, trues = self.prepare_metric()
            metric = list(self._metrics[0](preds, trues).items())[0][1]
            if metric > self._best_metric:
                self._best_metric = metric
                is_better = True
        return is_better

    # may need overwrite
    def prepare_metric(self):
        return np.concatenate(self._preds, axis=0), np.concatenate(self._trues, axis=0)


class DefaultModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super(DefaultModule, self).__init__(*args, **kwargs)

    def on_x_end(self, x, step):
        cost = None
        if len(self._losses) > 1:
            cost = collections.OrderedDict()
            for k,v in self._loss_dict.items():
                v = np.mean(v)
                cost.update({k:v})
                if hasattr(self._exp, '_writer') and self._exp._writer is not None:
                   self._exp._writer.add_scalar('loss/{}/{}/{}'.format('train' if self.training else 'valid',x,k),v,step)
        metrics = None
        if self.training:
            if self._metric_dict is not None:
                metrics = collections.OrderedDict()
                for k, v in self._metric_dict.items():
                    v = np.mean(v)
                    metrics.update({k:v})
                if hasattr(self._exp, '_writer') and self._exp._writer is not None:
                    for k,v in metrics.items():
                        self._exp._writer.add_scalar('metric/train/{}/{}'.format(x,k),v,step)
        else:
            preds, trues = self.prepare_metric()
            if x == 'epoch' and hasattr(self._exp, '_webviz') and self._exp._webviz is not None:
                index = np.arange(len(preds))
                Xpred = np.stack((index,preds),axis=-1)
                Xtrue = np.stack((index,trues),axis=-1)
                self._exp._webviz.scatter('hit-scatter',X=Xpred,name='pred',update='replace',opts=dict(showlegend=True,markersymbol='x',markersize=5))
                self._exp._webviz.scatter('hit-scatter',X=Xtrue,name='true',update='replace',opts=dict(markersymbol='dot',markersize=3))
            metrics= self.metrics(preds, trues)
            if metrics is not None:
                if hasattr(self._exp, '_writer') and self._exp._writer is not None:
                    for k,v in metrics.items():
                        self._exp._writer.add_scalar('metric/valid/{}/{}'.format(x,k),v,step)
        return cost, metrics
