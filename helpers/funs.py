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
import sys

import sys
from timeit import default_timer as timer
from datetime import datetime
from .. import colors
import numpy as np
import math
import inspect
from contextlib import contextmanager

def typecheck(**dwargs):
    dkeys = dwargs.keys()
    def _typecheck(fun):
        def _check(*args, **kwargs):
            signature = inspect.signature(fun)
            items = list(signature.parameters.items())
            for idx, arg in enumerate(args):
                kwargs[items[idx][0]] = arg
            for key, value in kwargs.items():
                if value is not None and key in dkeys:
                    dvalues = dwargs[key]
                    if not isinstance(dvalues, list):
                        dvalues = [dvalues]
                    matched = False
                    tvalue = type(value)
                    for dvalue in dvalues:
                        if dvalue == tvalue:
                            matched = True
                            break
                    if not matched:
                        raise TypeError('Type for `{}` not match.'
                                        '{} required, given {}'
                                        .format(key,
                                                colors.blue(dwargs[key]),
                                                colors.red(type(value))))
            return fun(**kwargs)
        return _check
    return _typecheck


def timestamp(date=None, fmt='%Y-%m-%d %H:%M:%S', split='-'):
    """ time stamp function
        Attributes
        ==========
            date : string / None
                   if date is None, return current time stamp as string
            fmt : string
                  datetime format to parse `date` or to format current time
                  see https://docs.python.org/3.6/library/datetime.html/strftime-strptime-behavior
                  for more description on format
            split : char / string
                    char or string to replace whitespace and ':' in datetime string
                    For example
                    > split = '-'
                    > now = 2018-03-29 12:00:00
                    > output: 2018-03-29-12-00-00

        Returns
        ==========
            datetime if date is not None, otherwise current timestamp as string

        Raises:
        ==========
            ValueError if date is not None and fmt incomplete or ambiguous
    """
    if date is None:
        now = datetime.now()
        if split is not None:
            fmt = fmt.replace(' ', split).replace(':', split)
        fmt = '{{0:{}}}'.format(fmt)
        return fmt.format(now)
    else:
        return datetime.strptime(date, fmt)


def stampit(targets, date=None, fmt='%Y%m%d%H%M%S', message=None, split=None, verbose=True):
    ''' add time stamp to string
        targets has form of {string: position}
        typically, string is a path, and position indicates the position to insert time stamp
        e.g., {'/path/to/your/string': 2} will become:
              {'/path/timestamp/to/your/string'}
    '''
    if not isinstance(targets, dict):
        raise TypeError('`targets` for stampit must be dict. given {}'.format(type(targets)))
    def _stampit(fun):
        def _wrap(*args, **kwargs):
            ts = timestamp(date, fmt, split)
            if message is not None:
                if isinstance(message, str):
                    ts = '{}_{}'.format(ts, message)
                else:
                    raise TypeError('`message` for stampit must be str or None. given {}'
                                    .format(colors.red(type(message))))
            if verbose:
                print('{}STAMP{}: => {}'
                      .format(colors.fg.blue,
                              colors.reset,
                              colors.red(ts)))
            signature = inspect.signature(fun)
            items = list(signature.parameters.items())
            for idx, arg in enumerate(args):
                kwargs[items[idx][0]] = arg
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if key in targets.keys():
                        splits = value.split('/')
                        pos = targets[key]
                        npos = pos
                        if npos < 0:
                            npos = len(splits) + pos
                        if len(splits) <= npos or npos < 0:
                            raise ValueError('npos `{}({})` must be shorter than splits({})'.format(npos, pos, len(splits)))
                        if splits[npos] == '':
                            splits[npos] = ts
                            if npos == 0:
                                splits.insert(0, '')
                            elif (npos+1) == len(splits):
                                splits.insert(npos+1, '')
                        else:
                            splits.insert(npos, ts)
                        kwargs[key] = '/'.join(splits)
                elif value is not None:
                    raise TypeError('type of `{}` must be str or None, given {}'.format(key, colors.red(type(value))))
            return fun(**kwargs)
        return _wrap
    return _stampit

<<<<<<< HEAD
def set_term_title(title):
    sys.stdout.write('\x1b]2;{}\x07'.format(title))

=======

def set_term_title(title):
    sys.stdout.write('\x1b]2;{}\x07'.format(title))


>>>>>>> 4e79866044983f5c23842fdffbc02413ebacbf5a
# @typecheck(x=int)
def intsize(x, cminus=False):
    if x > 0:
        return int(np.log10(x)) + 1
    elif x == 0:
        return 1
    else:
        if cminus:
            return int(np.log10(-x)) + 2
        else:
            return int(np.log10(-x)) + 1


def arg2dict(args, includes=None):
    kwargs = {}
    for arg in vars(args):
        if includes is None or arg in includes:
            kwargs[arg] = getattr(args, arg)
    return kwargs


def print_args(args, argcolor='green', valuecolor='red'):
    """ print arguments in the form of
        arg : value
        Attributes
        ==========
            args : arguments
                   arguments object returned by
                   argparser.ArgumentParsers().parse_args()
            argcolor : string / None
                       arg color. None will make no color
            valuecolor : string / None
                         value color. None will make no color
    """
    if argcolor is None:
        argcolor = lambda x:x
    else:
        argcolor = eval('colors.{}'.format(argcolor))
    if valuecolor is None:
        valuecolor = lambda x:x
    else:
        valuecolor = eval('colors.{}'.format(valuecolor))

    for arg in vars(args):
        print('{}: {}'.format(argcolor(arg), valuecolor(getattr(args, arg))))


def timeit(verbose=True):
    def _timeit(fun):
        def _wrap(*args, **kwargs):
            beg = timer()
            ret = fun(*args, **kwargs)
            end = timer()
            if verbose:
                print('Time elapsed for function {}: {}(s)'
                      .format(colors.blue(fun.__name__),
                              colors.red(end-beg)))
            return ret
        return _wrap
    return _timeit


@contextmanager
def time(message=None, color='blue'):
    if message is None:
        message = ''
    beg = timer()
    yield beg
    end = timer()
    print('>>> {}{}  <<<'.format(message, eval('colors.{}(end-bed)'.format(color))))


def line(iterable,
         epochs,
         iterations=None,
         brief=True,
         nprompts=10,
         feedbacks=False,
         timeit=False,
         accuracy=5,
         message=None,
         nc='x',
         cc='+'):
    """ Python Generator Tips [next|send / yield]
        show line message in a programmatic interactive mode
        to interactivate with line, call generator in the following order:
        (global_idx, elem, epoch, iteration) = (generator)
        while True:
            # do some work here
            (global_idx, elem, epoch, iteration) = generator.send(value)
            if need_break:
                break
        That is, to interactivate next / send with yield in generator, do
        it in the following loop:
                next(generator) \
                                 V
                    -------------|
                    |                   |
                    |                  V
                  send        yield
                    ^                 |
                    |                  |
                    |_______|

        Attibutes
        =========
            iterable : list / tuple / np.ndarray / range / None
                       iterable object or function that can yield something
                       generally speaking, iterable have length of:
                       "iterations * epochs"
            epochs : int or None
                     total iterations
                     if None, will be assigned as 1
            iterations : int / None
                         iterations per each epoch
                         NOTE:
                             - iterable is None, it will be assigned as
                               iterable = range(iterations * epochs)
                               therefore, iterations can not be None
                             - iterable is not None
                               - iterations is None, it will be assigned as
                                 iterations = len(iterable) / epochs
            brief : bool
                    if true, show extra information in form of
                        `[current-iter / total-iter]`
            nprompts : int
                       how many prompt (`nc`/`cc`) to mimic the progress
            feedbacks : boolean
                        if true, yield data and waiting for feedbacks from caller
                        else only yield data
            timeit : bool
                     time each iteration or not
            show_total_time : bool
                              show the sum of each iteration time if true
            accuracy : int
                       specifying print format precision for time
            message : string or None
                      message to be shown at the begining of each line
            nc : char
                 character to show as running prompt
            cc : char
                 character to show as running and ran prompt for each iteration
        Returns
        ==========
            tuple of (global_index, iterable_element, epoch, iteration), where:
            global_index : int
                            global index for iterable
            epoch : int
            iteration : int
                        iteration inside epoch
    """
    if epochs is None:
        epochs = 1
    if iterable is None:
        if iterations is None:
            raise ValueError('`iterable` and `iterations` can not both be None')
        iterable_size = iterations * epochs
        iterable = range(iterable_size)
    else:
        try:
            iterable_size = len(iterable)
        except TypeError:
            raise TypeError('getting the length of `iterable` failed')
        # number of iterations per each epoch
        if iterations is None:
            iterations = int(iterable_size / epochs)
    epochsize = intsize(epochs)
    if message is None:
        message = '@'
    nc = nc.encode('utf8')
    cc = cc.encode('utf8')
    # rotation times per one `+` / `x`
    scale = nprompts / iterations
    step = max(1, int(scale))
    _prompt = np.asarray([nc] * nprompts, dtype=np.string_)
    itersize = intsize(iterations)
    beg = None
    elapsed = None
    if brief:
        if timeit:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:3}}%]' \
                   ' {{}} @{{:0>.{}f}}(s){}'.format(epochsize,
                                                   message,
                                                   epochs,
                                                   nprompts,
                                                   accuracy,
                                                   colors.red('$'))
            def format_message(epoch, prompt, iteration, iterations,
                               percentage, retvalue, elapsed_time):
                return spec.format(epoch, prompt, percentage,
                                   retvalue, elapsed_time)

        else:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:3}}%] {{}}{}' \
                   .format(epochsize,
                           message,
                           epochs,
                           nprompts,
                           colors.red('$'))
            def format_message(epoch, prompt, iteration, iterations,
                               percentage, retvalue, elapsed_time):
                return spec.format(epoch, prompt, percentage,retvalue)
    else:
        if timeit:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%]' \
                   ' {{}} @{{:0>.{}f}}(s){}'.format(epochsize,
                                                    message,
                                                    epochs,
                                                    nprompts,
                                                    itersize,
                                                    itersize,
                                                    accuracy,
                                                    colors.red('$'))
            def format_message(epoch, prompt, iteration, iterations,
                               percentage, retvalue, elapsed_time):
                return spec.format(epoch, prompt, iteration, iterations,
                                   percentage, retvalue, elapsed_time)
        else:
            spec = '\r<{{:<{}}}{}{}> [{{:{}}}, {{:{}}} / {{:{}}}, {{:3}}%]' \
                   ' {{}}{}'.format(epochsize,
                                    message,
                                    epochs,
                                    nprompts,
                                    itersize,
                                    itersize,
                                    colors.red('$'))
            def format_message(epoch, prompt, iteration, iterations,
                               percentage, retvalue, elapsed_time):
                return spec.format(epoch, prompt, iteration, iterations,
                                   percentage, retvalue)
    def _line():
        totaltime = 0
        prev = 0
        idx = 0
        while True:
            elem = iterable[idx % iterable_size]
            time_beg = timer()
            epoch = int(idx / iterations)
            iteration = (idx % iterations)
            if feedbacks:
                ret = (yield (idx, elem, epoch, iteration))
            else:
                yield (idx, elem, epoch, iteration)
                ret = ''
            # convert iteration index to nprompts index
            block_beg = int(iteration * scale)
            if block_beg >= nprompts:
                block_beg = nprompts - step
            block_end = block_beg + step
            elapsed = round((timer() - time_beg), accuracy)
            totaltime += elapsed
            iteration += 1

            percentage = int(iteration * 100 / iterations)

            if _prompt[block_beg] == nc or prev != block_beg or percentage == 100:
                _prompt[block_beg:block_end] = cc
                if prev != block_beg:
                    _prompt[prev:block_beg] = cc
            else:
                _prompt[block_beg:block_end] = nc
            prev = block_beg
            prompt = _prompt[:block_end].tostring().decode('utf-8')
            end_flag = ''
            if percentage == 100:
                # start new epoch
                end_flag = '\n'
                elapsed = totaltime
                totaltime = 0
            message = format_message(epoch+1, prompt, iteration, iterations,
                                     percentage, ret, elapsed)
            print(message, end=end_flag, flush=True)
            idx += 1
        print()
    return _line, iterable_size


def hitmap(predict, label, nclass):
    if not isinstance(predict, np.ndarray):
        raise TypeError('`predict` must be np.ndarray, given {}'
                        .format(type(predict)))
    if not isinstance(label, np.ndarray):
        raise TypeError('`label` must be np.ndarray, given {}'
                        .format(type(label)))
    if len(predict.shape) == 2:
        predict = np.argmax(predict, axis=1)
    elif len(predict.shape) > 2:
        raise ValueError('`predict` shape must be 1/2, given {}'.format(predict.shape))
    if len(label.shape) == 2:
        label = np.argmax(label, axis=1)
    elif len(label.shape) > 2:
        raise ValueError('`label` shape must be 1/2, given {}'.format(label.shape))

    # predict / label: [batch-size]
    matrix = np.zeros(shape=(nclass, nclass))
    for p, l in zip(predict, label):
        matrix[p, l] += 1
    return matrix
