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
from .. import helpers
from . import helper
from . import core

def base_pool2d(input_shape, fun, pshape, stride,
                padding, reuse, name, scope):
    helper.check_input_shape(input_shape)
    if stride is None:
        stride = pshape
    stride = helper.norm_input_2d(stride)
    pshape = helper.norm_input_2d(pshape)
    out_shape = helper.get_output_shape(input_shape,
                                        input_shape[core.axis],
                                        pshape,
                                        stride,
                                        padding)

    ltype = fun.__name__.rsplit('.', 1)
    if len(ltype) > 1:
        ltype = ltype[1]
    else:
        ltype = ltype[0]
    ops_scope, _, name = helper.assign_scope(name, scope, ltype, reuse)
    def _base_pool2d(x):
        with ops_scope:
            return fun(x, pshape, stride, padding.upper(),
                       data_format=core.data_format,
                       name=name)
    return _base_pool2d, out_shape


def base_pool2d_global(input_shape, fun, reuse, name, scope):
    # if none and e.g., fun.__name__ == 'max_pool'
    #    name = max_pool
    helper.check_input_shape(input_shape)
    ltype = fun.__name__.rsplit('.', 1)
    if len(ltype) > 1:
        ltype = ltype[1]
    else:
        ltype = ltype[0]
    ops_scope, _, name = helper.assign_scope(name, scope, ltype, reuse)
    axes = [idx for idx, _ in enumerate(input_shape)]
    del axes[core.axis]
    del axes[0]
    def _base_pool2d_global(x):
        with ops_scope:
            return fun(x, axis=axes, name=name)
    return _base_pool2d_global, [input_shape[0], input_shape[core.axis]]


@helpers.typecheck(input_shape=list,
                   pshape=[int, list],
                   stride=[int, list],
                   padding=str,
                   reuse=bool,
                   name=str,
                   scope=str)
def avg_pool2d(input_shape,
               pshape=2,
               stride=None,
               padding='same',
               reuse=False,
               name=None,
               scope=None):
    return base_pool2d(input_shape, core.avg_pool, pshape,
                       stride, padding, reuse, name, scope)


@helpers.typecheck(input_shape=list,
                   reuse=bool,
                   name=str,
                   scope=str)
def avg_pool2d_global(input_shape,
                      reuse=False,
                      name=None,
                      scope=None):
    return base_pool2d_global(input_shape, core.mean,
                              reuse, name, scope)


@helpers.typecheck(input_shape=list,
                   pshape=[int, list],
                   stride=[int, list],
                   padding=str,
                   reuse=bool,
                   name=str,
                   scope=str)
def max_pool2d(input_shape,
               pshape=2,
               stride=None,
               padding='same',
               reuse=False,
               name=None,
               scope=None):
    return base_pool2d(input_shape, core.max_pool, pshape,
                       stride, padding, reuse, name, scope)


@helpers.typecheck(input_shape=list,
                   reuse=bool,
                   name=str,
                   scope=str)
def max_pool2d_global(inputs,
                      reuse=False,
                      name=None,
                      scope=None):
    return base_pool2d_global(input_shape, core.max,
                              reuse, name, scope)


@helpers.typecheck(input_shape=list,
                   output_shape=list,
                   factor=[int, float],
                   mode=str,
                   align_corners=bool,
                   reuse=bool,
                   name=str,
                   scope=str)
def resize(input_shape,
           output_shape=None,
           factor=None,
           mode='bilinear',
           align_corners=False,
           reuse=False,
           name=None,
           scope=None):
    helper.check_input_shape(input_shape)
    if output_shape is None:
        if factor is None:
            raise ValueError('cannot feed bilinear with both '
                             'none of output_shape and factor')
        output_shape = [x for x in input_shape]
        if isinstance(factor, (int, float)):
            factor = [1] + [factor] * (len(output_shape)-2) + [1]
        elif isinstance(factor, (list, tuple)):
            if len(factor) == 1:
                factor = [1] + factor * (len(output_shape) - 2) + [1]
            elif len(factor) == (len(output_shape) - 2):
                factor = [1] + factor + [1]
            elif len(factor) != len(output_shape):
                raise ValueError('factor and output_shape '
                                 'length not match. {} vs {}'
                                 .format(factor, output_shape))
        else:
            raise TypeError('factor type not support in bilinear')
        for i in range(1, len(output_shape)-1):
            output_shape[i] *= factor[i]
    if mode not in ['bilinear', 'bicubic', 'area', 'nearest_neighbor']:
        raise ValueError('mode must be one of '
                         '[bilinear, bicubic, area, nearest_neighbor]')
    ops_scope, _, name = helper.assign_scope(name, scope, model, reuse)
    def _resize(x):
        with ops_scope:
            return eval('core.resize_{}(x, output_shape, align_corners, name)'
                        .format(mode))
    return _resize
