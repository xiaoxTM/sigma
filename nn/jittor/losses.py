#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Renwu Gao
@Contact: re.gao@szu.edu.cn
@File: losses.py
@Time: 03/15/20 16:43 PM
"""

import argparse
import numpy as np
import random
import jittor as jt
from jittor import nn
import os
import functools
from .utils import one_hot
from sigma import parse_params, expand_params
from sigma.fontstyles import colors


def margin(caxis=1,weights=1.0,mpos=0.9,mneg=0.1,alpha=0.5,reduction='mean'):
    def _margin_loss(preds, trues):
        nclass = preds.size(caxis)
        t = one_hot(trues, nclass)
        loss = t.float() * jt.relu(mpos-preds).pow(2) + alpha * (1.0-t.float()) * jt.relu(preds-mneg).pow(2)
        loss = loss * weights
        return {'margin':loss.mean() if reduction == 'mean' else loss.sum()} # loss.sum() for milestone version
    return _margin_loss

def smoothed_cce(caxis=1, eps=0.2, reduction='mean'):
    def _smoothed_cce(preds, trues):
        nclass = preds.size(caxis)
        t = one_hot(trues, nclass).float()
        t = t * (1.0-eps) + (1.0-t) * eps / (nclass-1.0)
        log_prob = nn.log_softmax(preds, dim=1)
        loss = -(t * log_prob).sum(dim=1)
        return {'scce':loss.mean() if reduction=='mean' else loss.sum()}
    return _smoothed_cce

def l1(*args, **kwargs):
    loss = nn.L1Loss(*args, **kwargs)
    def _l1(preds, trues):
        return {'l1':loss(preds,trues)}
    return _l1

def mse(*args, **kwargs):
    loss = nn.MSELoss(*args, **kwargs)
    def _mse(preds, trues):
        return {'mse':loss(preds,trues)}
    return _mse

def cce(*args, **kwargs):
    loss = nn.CrossEntropyLoss(*args, **kwargs)
    def _cce(preds,trues):
        return {'cce':loss(preds,trues)}
    return _cce

#def ctc(*args, **kwargs):
#    loss = nn.CTCLoss(*args, **kwargs)
#    def _ctc(preds, trues):
#        return {'ctc':loss(preds,trues)}
#    return _ctc
#
def nll(*args, **kwargs):
    loss = functools.partial(jittor.nll_loss,*args,**kwargs)
    def _nll(preds, trues):
        return {'nll':loss(preds, trues)}
    return _nll

#def poisson(*args, **kwargs):
#    loss = nn.PoissonNLLLoss(*args, **kwargs)
#    def _poisson(preds, trues):
#        return {'pos':loss(preds, trues)}
#    return _poisson
#
#def kldiv(*args, **kwargs):
#    loss = nn.KLDivLoss(*args, **kwargs)
#    def _kldiv(preds, trues):
#        return {'kld':loss(preds, trues)}
#    return _kldiv

def bce(*args, **kwargs):
    loss = nn.BCELoss(*args, **kwargs)
    def _bce(preds, trues):
        return {'bce':loss(preds, trues)}
    return _bce

def lbce(*args, **kwargs):
    loss = nn.BCEWithLogitsLoss(*args, **kwargs)
    def _lbce(preds, trues):
        return {'lbce':loss(preds, trues)}
    return _lbce

#def ranking(*args, **kwargs):
#    loss = nn.MarginRankingLoss(*args, **kwargs)
#    def _ranking(preds, trues):
#        return {'ranking':loss(preds, trues)}
#    return _ranking
#
#def hinge(*args, **kwargs):
#    loss = nn.HingeEmbeddingLoss(*args, **kwargs)
#    def _hinge(preds, trues):
#        return {'hinge':loss(preds,trues)}
#    return _hinge
#
#def xlmargin(*args, **kwargs):
#    loss = nn.MultiLabelSoftMarginLoss(*args, **kwargs)
#    def _xlmargin(preds, trues):
#        return {'xlmn':loss(preds, trues)}
#    return _xlmargin

def smoothl1(reduction='mean'):
    loss = functools.partial(nn.smooth_l1_loss,reduction=reduction)
    def _smoothl1(preds, trues):
        return {'sl1':loss(trues, preds)}
    return _smoothl1

#def softmargin(*args, **kwargs):
#    loss = nn.SoftMarginLoss(*args, **kwargs)
#    def _softmargin(preds, trues):
#        return {'sm':loss(preds, trues)}
#    return _softmargin
#
#def xlsoftmargin(*args, **kwargs):
#    loss = nn.MultiLabelSoftMarginLoss(*args, **kwargs)
#    def _xlsoftmargin(preds, trues):
#        return {'xlsm':loss(preds, trues)}
#    return _xlsoftmargin
#
#def cosine(*args, **kwargs):
#    loss = nn.CosineEmbeddingLoss(*args, **kwargs)
#    def _cosine(preds, trues):
#        return {'cos':loss(preds,trues)}
#    return _cosine
#
#def xmargin(*args, **kwargs):
#    loss = nn.MultiMarginLoss(*args, **kwargs)
#    def _xmargin(preds, trues):
#        return {'xm':loss(preds,trues)}
#    return _xmargin
#
#def tripletmargin(*args, **kwargs):
#    loss = nn.TripletMarginLoss(*args, **kwargs)
#    def _tripletmargin(preds, trues):
#        return {'tm':loss(preds, trues)}
#    return _tripletmargin


__losses__ = { 'l1': l1,
               'mse': mse,
               'l2': mse,
               'cce': cce, # cce = softmax -> log -> nll
               'scce': smoothed_cce,
               'nll': nll,
               'bce': bce,
               'lbce': lbce, # lbce = softmax -> bce
               'smoothl1': smoothl1}


def get(loss):
    if loss is None or callable(loss):
        return loss
    elif isinstance(loss, str):
        loss = loss.trip()
        loss_type, params = parse_params(loss)
        loss_type = loss_type.lower()
        assert loss_type in __losses__.keys(), '{} loss type not support'.format(loss_type)
        return __losses__[loss_type](**params)
    else:
        raise TypeError('cannot convert type {} into Loss'.format(colors.red(type(loss))))


def register(key, loss):
    assert key is not None and loss is not None, 'both key and loss can not be None'
    global __losses__
    assert key not in __losses__.keys(), 'loss `{}` already exists'.format(key)
    assert isinstance(loss, _Loss) or callable(loss), 'loss must be either an instance of _Loss or function, given {}'.format(type(loss))
    __losses__.update({key:loss})
