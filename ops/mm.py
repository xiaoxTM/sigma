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

from . import initializers
from . import regularizers
from . import core

def malloc(name,
           layername,
           shape,
           dtype=None,
           initializer=None,
           regularizer=None,
           cpuid=0,
           trainable=True,
           collections=None, # default is GraphKeys.GLOBAL_VARIABLES
           summary='histogram',
           reuse=False,
           scope=None,
           **kwargs):
    if name is None or layername is None:
        raise ValueError('`name` or `layername` not given.')
    initializer = initializers.get(initializer)
    regularizer = regularizers.get(regularizer)
    add_to_collect = True
    if scope is None:
        variable_scope = layername
        add_to_collect = False
    else:
        variable_scope = '{}/{}'.format(scope, layername)
<<<<<<< HEAD
    #variable_type = 'trainable'
    #if not trainable:
    #    variable_type = 'non-trainable'
    #variable_scope = '{}/variables/{}'.format(variable_scope, variable_type)
=======
    # since variable type will cause pretrain loading problem
    # remove it
    # variable_type = 'trainable'
    # if not trainable:
    #     variable_type = 'non-trainable'
>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a
    variable_scope = '{}/variables'.format(variable_scope)
    with core.variable_scope(variable_scope, reuse=reuse):
        variable = core.get_variable(name, shape, dtype, initializer,
                                     regularizer, trainable, collections)
    if add_to_collect and not reuse:
        core.add_to_collection(scope, variable)
    if summary is not None:
        with core.device('/cpu:{}'.format(cpuid)):
            core.summarize(variable.name, variable, summary, norm=False, reuse=reuse)
    return variable


def local_variable(name,
                   layername,
                   shape,
                   dtype=None,
                   initializers=None,
                   regularizer=None,
                   cpuid=0,
                   trainable=True,
                   summary='histogram',
                   reuse=False,
                   scope=None,
                   **kwargs):
    return malloc(name,
                  layername,
                  shape,
                  dtype,
                  initializer,
                  regularizer,
                  cpuid,
                  trainable,
                  core.Collections.local_variables,
                  summary,
                  reuse,
                  scope,
                  **kwargs)


def global_variable(name,
                    layername,
                    shape,
                    dtype=None,
                    initializers=None,
                    regularizer=None,
                    cpuid=0,
                    trainable=True,
                    summary='histogram',
                    reuse=False,
                    scope=None,
                    **kwargs):
    return malloc(name,
                  layername,
                  shape,
                  dtype,
                  initializer,
                  regularizer,
                  cpuid,
                  trainable,
                  core.Collections.global_variables,
                  summary,
                  reuse,
                  scope,
                  **kwargs)
