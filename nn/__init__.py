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

import traceback
import logging
# from sigma.fontstyles import colors

caxis = -1 # channel axis
floatx = 'float32'
data_format = ['NC','NWC','NHWC','NDHWC']

def get():
    config = {}
    try:
        import sigma.nn.pytorch
        conf = sigma.nn.pytorch.get()
        if conf is not None:
            config['torch'] = conf
        config['backend'] = 'torch'
    except Exception as e:
        logging.warning('torch module disabled because:', e)
        logging.warning(traceback.print_exc())
        logging.warning('trying jittor')
        try:
            import sigma.nn.pyjittor
            conf = sigma.nn.pyjittor.get()
            if conf is not None:
                config['jittor'] = conf
            config['backend'] = 'jittor'
        except Exception as e:
            logging.warning('jittor module disabled because:', e)
            logging.warning(traceback.print_exc())
    config['floatx'] = floatx
    config['data_format'] = data_format
    return config


def set(config):
    backend = config.get('backend', 'None')
    from sigma.fontstyles import colors
    if backend == 'torch':
        try:
            import sigma.nn.pytorch
            conf = config.get('torch', None)
            if conf is not None:
                sigma.nn.torch.set(conf)
            logging.info('using {}<{}> backend'.format(colors.red(backend),colors.green(sigma.nn.pytorch.__version__)))
        except Exception as e:
            logging.warning('torch backend disabled because:', e)
            logging.warning(traceback.print_exc())
    elif backend == 'jittor':
        try:
            import sigma.nn.pyjittor
            conf = config.get('jittor', None)
            if conf is not None:
                sigma.nn.pyjittor.set(conf)
            logging.info('using {}<{}> backend'.format(colors.red(backend),colors.green(sigma.nn.pyjittor.__version__)))
        except Exception as e:
            logging.warning('jittor backend disabled because:', e)
            logging.warning(traceback.print_exc())
    else:
        logging.warning(colors.red('no backend specified'))
    global caxis
    global data_format
    if data_format[-1][1] == 'C':
        caxis = 1
    elif data_format[-1][-1] == 'C':
        caxis = -1
    global floatx
    floatx = config.get('floatx', 'float32')
