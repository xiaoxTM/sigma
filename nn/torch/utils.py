#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Renwu Gao
@Contact: re.gao@szu.edu.cn
@File: utils.py
@Time: 03/15/20 16:43 PM
"""

import sigma
from sigma import version as vsn
from sigma import metrics as met
from sigma.nn.torch import activations as acts
from sigma.nn.torch import normalizations as norms
from sigma.nn.torch import initializers
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import os


def count_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    return total / 1e6


def set_seed(seed=1024, deterministic=False):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic is True:
            torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=deterministic # cudnn
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = '{}'.format(seed)

def seed_workers(seed):
    def _seed_workers(worker_id):
        np.random.seed(seed+worker_id)
    return _seed_workers

@sigma.defaultable
def build_conv3d(cin, couts,
                 ksize=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 act='relu(inplace=True)',
                 norm='bn3d',
                 weight_initializer=None,
                 bias_initializer=None,
                 dropout=None):
    x = nn.Sequential()
    if not isinstance(couts, (list, tuple)):
        couts = [couts]
    ksize,stride,padding,dilation,groups,bias,padding_mode,act,norm,dropout = \
        sigma.expand_params([ksize,stride,padding,dilation,groups,bias,padding_mode,act,norm,dropout],
                            len(couts))
    weight_initializer = initializers.get(weight_initializer)
    bias_initializer = initializers.get(bias_initializer)
    for idx, cout in enumerate(couts):
        conv = nn.Conv3d(cin,cout,
                         kernel_size=ksize[idx],
                         stride=stride[idx],
                         padding=padding[idx],
                         dilation=dilation[idx],
                         groups=groups[idx],
                         bias=bias[idx],
                         padding_mode=padding_mode[idx])
        if weight_initializer is not None:
            weight_initializer(conv.weight.data)
        if bias[idx] and bias_initializer is not None:
            bias_initializer(conv.bias.data)
        x.add_module('conv{}'.format(idx), conv)
        if norms[idx] is not None:
            x.add_module('norm{}'.format(idx),norms.get(norms[idx],cout,'3d'))
        if act[idx] is not None:
            x.add_module('act{}'.format(idx), acts.get(act[idx]))
        if dropout[idx] is not None:
            x.add_module('dropout{}'.format(idx),nn.Dropout(dropout[idx]))
        cin = cout
    return x


@sigma.defaultable
def build_conv2d(cin, couts,
                 ksize=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 act='relu(inplace=True)',
                 norm='bn2d',
                 weight_initializer=None,
                 bias_initializer=None,
                 dropout=None):
    x = nn.Sequential()
    if not isinstance(couts, (list, tuple)):
        couts = [couts]
    ksize,stride,padding,dilation,groups,bias,padding_mode,act,norm,dropout=\
        sigma.expand_params([ksize,stride,padding,dilation,groups,bias,padding_mode,act,norm,dropout],
                            len(couts))
    weight_initializer = initializers.get(weight_initializer)
    bias_initializer = initializers.get(bias_initializer)
    for idx, cout in enumerate(couts):
        conv = nn.Conv2d(cin,cout,
                         kernel_size=ksize[idx],
                         stride=stride[idx],
                         padding=padding[idx],
                         dilation=dilation[idx],
                         groups=groups[idx],
                         bias=bias[idx],
                         padding_mode=padding_mode[idx])
        if weight_initializer is not None:
            weight_initializer(conv.weight.data)
        if bias[idx] and bias_initializer is not None:
            bias_initializer(conv.bias.data)
        x.add_module('conv{}'.format(idx), conv)
        if norm[idx] is not None:
            x.add_module('norm{}'.format(idx),norms.get(norm[idx],cout,'2d'))
        if act[idx] is not None:
            x.add_module('act{}'.format(idx),acts.get(act[idx]))
        if dropout[idx] is not None:
            x.add_module('dropout{}'.format(idx),nn.Dropout(dropout[idx]))
        cin = cout
    return x


@sigma.defaultable
def build_conv1d(cin, couts,
                 ksize=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 act='relu(inplace=True)',
                 norm='bn1d',
                 weight_initializer=None,
                 bias_initializer=None,
                 dropout=None):
    x = nn.Sequential()
    if not isinstance(couts, (list, tuple)):
        couts = [couts]
    ksize,stride,padding,dilation,groups,bias,padding_mode,act,norm,dropout=\
        sigma.expand_params([ksize,stride,padding,dilation,groups,bias,padding_mode,act,norm,dropout],
                            len(couts))
    weight_initializer = initializers.get(weight_initializer)
    bias_initializer = initializers.get(bias_initializer)
    for idx, cout in enumerate(couts):
        conv = nn.Conv1d(cin,cout,
                         kernel_size=ksize[idx],
                         stride=stride[idx],
                         padding=padding[idx],
                         dilation=dilation[idx],
                         groups=groups[idx],
                         bias=bias[idx],
                         padding_mode=padding_mode[idx])
        if weight_initializer is not None:
            weight_initializer(conv.weight.data)
        if bias[idx] and bias_initializer is not None:
            bias_initializer(conv.bias.data)
        x.add_module('conv{}'.format(idx), conv)
        if norm[idx] is not None:
            x.add_module('norm{}'.format(idx),norms.get(norm[idx],cout,'1d'))
        if act[idx] is not None:
            x.add_module('act{}'.format(idx),acts.get(act[idx]))
        if dropout[idx] is not None:
            x.add_module('dropout{}'.format(idx),nn.Dropout(dropout[idx]))
        cin = cout
    return x


@sigma.defaultable
def build_linear(cin,couts,
                 bias=True,
                 act='relu(inplace=True)',
                 norm='bn1d',
                 weight_initializer=None,
                 bias_initializer=None,
                 dropout=None):
    x = nn.Sequential()
    if not isinstance(couts, (list, tuple)):
        couts = [couts]
    bias,act,norm,dropout=sigma.expand_params([bias,act,norm,dropout],len(couts))
    weight_initializer = initializers.get(weight_initializer)
    bias_initializer = initializers.get(bias_initializer)
    for idx, cout in enumerate(couts):
        conv = nn.Linear(cin,cout,bias=bias[idx])
        if weight_initializer is not None:
            weight_initializer(conv.weight.data)
        if bias[idx] and bias_initializer is not None:
            bias_initializer(conv.bias.data)
        x.add_module('conv{}'.format(idx),conv)
        if norm[idx] is not None:
            x.add_module('norm{}'.format(idx),norms.get(norm[idx],cout,'1d'))
        if act[idx] is not None:
            x.add_module('act{}'.format(idx), acts.get(act[idx]))
        if dropout[idx] is not None:
            x.add_module('dp{}'.format(idx), nn.Dropout(p=dropout[idx]))
        cin = cout
    return x


def one_hot(x, nclass):
    # x: [batch-size]
    if vsn.lt(torch.__version__, '1.1.0'):
        device = torch.device('cuda:{}'.format(x.get_device()))
        t = torch.zeros(x.size(0), nclass, device=device).long()
        t = t.scatter_(1, x.data.view(-1, 1), 1)
        t = Variable(t)
    else:
        t = F.one_hot(x, nclass)
    return t


"""
    get the indices of k nearest neighbours
    accorrding to the Euclidean distance
    and then calculate the distance between
    some center and its k neighbors
    x : [batch-size, dims, num-points]
"""
@torch.no_grad()
def knn(x, k, mode='euc'):
    assert mode in ['euc', 'dot']
    # x: [batch-size, dim, num-points]
    assert k < x.size(2), 'k should be small than number of points, given [k:{}/num-points:{}]'.format(k, x.size(2))
    # inner: [batch-size, num-points, num-points]
    pairwise_distance = torch.matmul(x.transpose(2,1),x)
    if mode == 'euc':
        inner = -2*pairwise_distance
        # y: [batch-size, 1, num-points]
        y = torch.sum(x**2, dim=1, keepdim=True)
        # pairwise-distance: [batch-size, num-points, num-points]
        pairwise_distance = -y - inner - y.transpose(2, 1)
    # return: [batch-size, num-points, k]
    return pairwise_distance.topk(k=k, dim=2)


def norm(x, dim=1, keepdim=False, epsilon=1e-9):
    squared = x.pow(2).sum(dim=dim, keepdim=keepdim) + epsilon
    return squared.sqrt()


def squash(x, dim=1, epsilon=1e-9):
    squared = x.pow(2).sum(dim=dim, keepdim=True) + epsilon
    sqrt = squared.sqrt()
    x = x * (squared / (1 + squared) / sqrt)
    return x


def dynamic_routing(x, bias=0, n_iteration=3):
    batch_size, outdims, outcaps, incaps = x.size()
    b = torch.zeros((batch_size, 1, outcaps, incaps)).cuda()

    for n_iter in range(n_iteration):
        c = F.softmax(b, dim=2)
        if (n_iter+1) == n_iteration:
            s = (c * x).sum(dim=3, keepdim=True) + bias
            v = squash(s)
        else:
            with torch.no_grad():
                s = (c * x).sum(dim=3, keepdim=True) + bias
                v = squash(s)
                b = b + (x * v).sum(dim=1, keepdim=True)
    v = v.squeeze(3)
    return v


def maskout(x, nclass, labels=None, drop=False):
    # x: [batch-size, dims, capsules]
    # labels: None or [batch-size]
    if labels is None:
        # preds: [batch-size, capsules]
        preds = norm(x)
        # labels: [batch-size]
        labels = preds.argmax(dim=1)
    # [batch-size, nclass]
    onehot = one_hot(labels, nclass)
    # [batch-size, 1, nclass]
    onehot = torch.unsqueeze(onehot, dim=1)
    if drop:
        batch_size, dims, _ = x.size()
        return torch.masked_select(x, onehot.eq(1)).view(batch_size, dims)
    return x * onehot.float()


if __name__ == '__main__':
    labels=np.random.randint(0, 4, size=3, dtype=np.int64)
    torch_labels = torch.from_numpy(labels)
    onehot = one_hot(torch_labels)
    print(onehot)
    print(one_hot(torch_labels, nclass=4))

    x = np.random.randn(3, 3, 4)
    torch_x = torch.from_numpy(x)
    m = maskout(torch_x, torch_labels, nclass=4)
    print('input:', x)
    print('maskout:', m)
    print('maskout with drop:', maskout(torch_x, torch_labels, nclass=4, drop=True))

    #points = np.random.rand(2,4,2)
    #extras = np.random.rand(2,2,2)
    #x = np.concatenate([points, extras], axis=1)
    #print('target:',points)
    #print('extras:', extras)
    #print('x:', x)
    #pool = chamfer_pool(torch.from_numpy(x), torch.from_numpy(points))
    #print('pooled diff:', torch.from_numpy(points)-pool)
