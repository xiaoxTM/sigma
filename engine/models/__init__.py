from __future__ import annotations

import typing

import torch
from torch import nn

import abc
# from torch.utils.tensorboard import SummaryWriter
import logging
from sigma.core.fontstyles import base as fontstylebase
from torch.nn.modules.module import T

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class SigmaModel(nn.Module, abc.ABC):
    def __init__(self,
                 input_size: list[int],
                 mixed_precision: bool,
                 hardware_platform: str,
                 freeze_batch_norm: bool) -> None:
        super(SigmaModel, self).__init__()
        self.freeze_batch_norm = freeze_batch_norm
        self.input_size = input_size
        self.hardware_platform = hardware_platform
        self.mixed_precision = mixed_precision
        self.tag = (f'{hardware_platform}-'
                    f'{input_size[0]}x{input_size[1]}-'
                    f'mp{mixed_precision}_fbn{freeze_batch_norm}')

    def train_epoch_begin(self, *args, **kwargs):
        pass

    def train_epoch_end(self, *args, **kwargs):
        pass

    def train_iter_begin(self, *args, **kwargs):
        pass

    def train_iter_end(self, *args, **kwargs):
        pass

    def preprocess(self,
                   data: typing.Dict[str, torch.Tensor] = None,
                   inputs: torch.Tensor = None,
                   **kwargs) -> typing.Union[None, torch.Tensor]:
        assert data is not None or inputs is not None, 'data or inputs must be given'
        maxv = 6.931471805599453 # math.log(1024)
        if data is None and inputs is not None:
            return 2 * torch.log(inputs + 1) / maxv - 1.0  # [-1, 1]
        data.update({'hist': (2 * torch.log(data['hist'] + 1) / maxv - 1.0)})
        return data['hist']

    def postprocess(self, data: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        pass

    def train(self: T, mode: bool = True) -> T:
        super(SigmaModel, self).train(mode)
        if mode is True and self.freeze_batch_norm:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def infer(self: T,
              hist: torch.Tensor,
              shape: typing.Tuple[int, int] = None,
              **kwargs) -> typing.Dict[str, torch.Tensor]:
        """
        :param inputs: input depth image
        :param shape: desired output shape in the order of [height, width]
        :param kwargs: additional arguments
        """
        inputs = self.preprocess(inputs=hist, **kwargs)
        ret = self.forward(inputs, **kwargs)
        ret = self._infer(ret)
        return ret

    @abc.abstractmethod
    def _infer(self,
               result: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, torch.Tensor]:
        pass

    def summary(self,
                writer: torch.utils.tensorboard.SummaryWriter,
                prefix: str,
                result: typing.Dict[str, torch.Tensor],
                data: typing.Dict[str, torch.Tensor],
                stat: dict[str | typing.Any, typing.Any],
                global_step: int,
                epoch: int,
                idx: int,
                kwargs: typing.Dict[str | typing.Any, typing.Any] = None) -> None:
        writer.add_scalars('{}/metric'.format(prefix), {fontstylebase.clear(k): v for k, v in stat.items()}, global_step=global_step)
        if kwargs is not None:
            for k, v in kwargs.items():
                writer.add_scalar('{}/{}'.format(prefix, fontstylebase.clear(k)), v, global_step=global_step)
