#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Renwu Gao
@Contact: re.gao@szu.edu.cn
@File: checkpoint.py
@Time: 06/16/20 08:50 PM
"""

import os
import os.path
import torch
from torch.nn.parallel import DistributedDataParallel
import logging
from sigma.fontstyles import colors


def load(path, model, optimizer, scheduler, epoch=None, mode='latest', map_location=None):
    assert(os.path.exists(path)) and os.path.isdir(path), 'No checkpoint found in {}'.format(colors.red(path))
    assert epoch is not None or mode  in ['latest', 'best'], 'You must specify epoch when loading from a given epoch'
    ckpt = load_checkpoint(path, epoch, mode, map_location)
    if model is not None:
        if isinstance(model, DistributedDataParallel):
            model.module.load_state_dict(ckpt.pop('model'))
        else:
            model.load_state_dict(ckpt.pop('model'))
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt.pop('optimizer'))
    if scheduler is not None and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt.pop('scheduler'))
    return ckpt


def load_checkpoint(path, epoch, mode='latest', map_location=None):
    if mode=='latest':
        f = os.path.join(path, 'latest.pth')
    elif mode=='best':
        f = os.path.join(path, 'best-performance.pth')
    else:
        f = os.path.join(path, 'epoch-{}.pth'.format(epoch))
    assert os.path.exists(f) and os.path.isfile(f), 'file {} not found'.format(colors.red(f))
    realpath = os.path.realpath(f)
    ckpt = torch.load(realpath, map_location=map_location)
    if mode in ['latest', 'best']:
        # epoch-${NUM}.pth
        epoch = int(realpath.rsplit('/', 1)[1].rsplit('-', 1)[1].split('.')[0])
    ckpt['begin-epoch'] = epoch
    return ckpt


def save(path, model, optimizer, scheduler, epoch, is_best=False, **kwargs):
    data = {}
    if isinstance(model, DistributedDataParallel):
        data['model'] = model.module.state_dict()
    else:
        data['model'] = model.state_dict()
    if optimizer is not None:
        data['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()
    data.update(kwargs)
    cwd = os.getcwd()
    parent = os.path.abspath(path)
    # change to checkpoint directory
    os.chdir(parent)
    f = 'epoch-{}.pth'.format(epoch)
    torch.save(data, f)
    latest = 'latest.pth'
    if os.path.exists(latest):
        os.remove(latest)
    # relative path soft link
    os.symlink(f,latest)
    if is_best:
        best = 'best-performance.pth'
        if os.path.exists(best):
            os.remove(best)
        os.symlink(f, best)
    # change back to working directory
    os.chdir(cwd)


def resume(path, model, optimizer, scheduler, epoch=None, mode='latest', map_location=None):
    logging.info('resuming states from {}'.format(colors.green(path)))
    return load(path, model, optimizer, scheduler, epoch, mode, map_location)


class CheckPointer:
    def __init__(self, path=None):
        self.path = path
        if path is not None:
            os.makedirs(path, exist_ok=True)

    def load(self, model, optimizer, scheduler, epoch=None, mode='latest', map_location=None):
        load(self.path, model, optimizer, scheduler, epoch, mode, map_location)

    def save(self, model, optimizer, scheduler, epoch, is_best=False, **kwargs):
        save(self.path, model, optimizer, scheduler, epoch, is_best, **kwargs)

    def save_best(self, model, optimizer, scheduler, epoch):
        data = {}
        if isinstance(model, DistributedDataParallel):
            data['model'] = model.module.state_dict()
        else:
            data['model'] = model.state_dict()
        if optimizer is not None:
            data['optimizer'] = optimizer.state_dict()
        if scheduler is not None:
            data['scheduler'] = scheduler.state_dict()
        cwd = os.getcwd()
        parent = os.path.abspath(self.path)
        os.chdir(parent)
        f = 'epoch-{}.pth'.format(epoch)
        if not os.path.exists(f):
            torch.save(data, f)
        best = 'best-performance.pth'
        if os.path.exists(best):
            os.remove(best)
        os.symlink(f, best)
        os.chdir(cwd)


    def resume(self, model, optimizer, scheduler, epoch=None, mode='latest', map_location=None):
        return resume(self.path, model, optimizer, scheduler, epoch, mode, map_location)
