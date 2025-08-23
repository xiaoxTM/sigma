from __future__ import annotations

import math
import typing
import re

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import amp
from torch.nn import parallel, DataParallel
from torch import distributed as dist

import os.path
import logging
import numpy as np

from sigma.engine import misc
from sigma.engine.misc import ios, pcio
from sigma.engine.models import SigmaModel
from sigma.core.reporters.progressbar import ProgressBar
from sigma.core.fontstyles import colors, styles, base as fontstylebase
from sigma.core.utils import utils

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


class Runner:
    def __init__(self, config: typing.Dict[str, typing.Any]) -> None:
        """
        Initializes the object with the given configuration.

        Args:
            config (typing.Dict): A dictionary containing the configuration settings.

        Returns:
            None
        """
        # 构造函数
        self._logfile = open(os.path.join(config['workspace'], 'info.log'), 'w')
        self._iters = 0
        self._start_epoch = 0
        self._min_key_value = None
        self._config = config
        key = config['train']['key']
        if key is None:
            key = config['train']['metrics'][0] if config['train']['metrics'] is not None else 'loss'
        self._min_key = key
        self._scaler = amp.GradScaler('cuda', enabled=config['mixed_precision'])
        self._datasets = self.build_dataloaders('distributed' == config['run_mode'],
                                                **config['dataloader'])
        freeze_batch_norm = config['train']['freeze_batch_norm']
        self._model = self.build_models(mixed_precision=config['mixed_precision'],
                                        freeze_batch_norm=freeze_batch_norm,
                                        **config['model'])
        if config['train'].get('freeze_layer_patterns', None) is not None:
            """
            Freeze layers based on the given patterns.

            Parameters:
                patterns (List[str]): A list of patterns to match layer names.
                                      For more information, see https://docs.python.org/3/library/re.html
            """
            misc.freeze_layers_by_name(self._model, config['train']['freeze_layer_patterns'])
        self._optimizer = self.build_optimizer(**config['train']['optimizer'])
        self._schedulers = self.build_schedulers(**config['train']['schedulers'])
        if config['train']['checkpoint'] is not None:
            self.load_checkpoint(**config['train']['checkpoint'])
            if config['train']['checkpoint']['path'] is not None and not freeze_batch_norm:
                self.log(logger.warning, colors.yellow('train with pretrained model and batch norm unfrozen. it is '
                                                       'suggested to train with frozen batch norm when pretrained model'
                                                       ' is used.'))
        else:
            self.log(logger.info, colors.yellow('training from scratch'))
            if freeze_batch_norm:
                self.log(logger.warning, colors.yellow('training from scratch with batch norm frozen is not '
                                                       'recommended. please consider to use a pretrained model '
                                                       'instead or set freeze_batch_norm to False'))
        self._teacher = None
        if config.get('teacher', None) is not None:
            self._teacher = self.build_models(mixed_precision=config['mixed_precision'],
                                              freeze_batch_norm=freeze_batch_norm,
                                              **config['teacher']['model'])
            self.log(logger.info,
                     colors.blue('teacher built and knowledge distillation enabled with {} mode'.format(
                         'online' if config['teacher'].get('online', False) else 'offline')))
            ckpt = config['teacher']['checkpoint']
            if ckpt is not None:
                if os.path.exists(ckpt):
                    self.load_checkpoint(model=self._teacher, **ckpt)
                    self.log(logger.info, 'teacher loaded from {}'.format(colors.green(ckpt)))
                else:
                    raise ValueError(colors.red('teacher checkpoint not exists: {}'.format(ckpt)))
            else:
                if not config['teacher'].get('online', False):
                    raise ValueError(colors.red('checkpoint must be given when online is False'))
            self._teacher.eval()
        self._losses = self.build_losses(config['train']['losses'])
        self._metrics = self.build_metrics(config['train']['metrics'])
        if config['train'].get('summary', False):
            self.log(logger.info, colors.blue('tensorboard summary enabled'))
            self._writer = SummaryWriter(os.path.join(config['workspace'], 'logs'))
        os.makedirs(os.path.join(os.path.join(config['workspace'], 'ckpts')), exist_ok=True)

    def __del__(self):
        # 析构函数
        if not self._logfile.closed:
            self._logfile.close()

    @property
    def model(self) -> typing.Union[nn.Module, None]:
        """
        Returns the model object of the class.

        Returns:
            Union[nn.Module, None]: The model object if it exists, otherwise None.
        """
        if not hasattr(self, '_model'):
            return None
        if self._config['run_mode'] in ['distributed', 'data-parallel']:
            return self._model.module
        return self._model

    @property
    def writer(self) -> typing.Union[SummaryWriter, None]:
        """
        Returns the SummaryWriter object if it exists, otherwise returns None.
        """
        if not hasattr(self, '_writer'):
            return None
        return self._writer

    def build_dataloaders(self, distributed: bool, **kwargs) -> typing.Dict[str, DataLoader]:
        """
        Builds and returns a dictionary of dataloaders.

        Parameters:
            distributed (bool): A boolean indicating whether to use distributed mode.
            semantic_segment (bool): A boolean indicating whether to use semantic segmentation.
            **kwargs: Additional keyword arguments to be passed to the dataloaders.get() function.

        Returns:
            typing.Dict[str, DataLoader]: A dictionary of dataloaders, where the keys are the names of the dataloaders and the values are DataLoader objects.
        """
        from sigma.engine import dataloaders
        self.log(logger.info, 'building dataloaders with distributed mode: {}'.format(colors.blue(distributed)))
        return dataloaders.get(distributed=distributed, **kwargs)

    def build_models(self, **kwargs) -> nn.Module:
        """
        Builds and returns a neural network model based on the given parameters.

        Args:
            **kwargs: Keyword arguments that specify the model parameters.

        Returns:
            nn.Module: The built neural network model.

        Raises:
            AssertionError: If torch.cuda is not available.
            AssertionError: If the built model is not inherited from VisionModel.

        Note:
            This function assumes that torch.cuda is available and imports the necessary modules from src.models.
            It also logs information about the built model and the mode of execution.

        """
        assert torch.cuda.is_available()
        model = arch.get(kwargs['name'],
                         mixed_precision=kwargs['mixed_precision'],
                         freeze_batch_norm=kwargs['freeze_batch_norm'],
                         **kwargs['params'])
        assert isinstance(model, SigmaModel), 'the built model is not inherited from SigmaModel'
        num_gpus = torch.cuda.device_count()
        num_params = misc.trainable_parameters(model)
        # if self.logable():
        self.log(logger.info,
                 'model {} built with {}M parameters in {} mode using {} GPUs'.format(colors.blue(model.tag),
                                                                                      colors.blue(num_params),
                                                                                      colors.blue(
                                                                                          self._config['run_mode']),
                                                                                      colors.blue(num_gpus)))
        if self._config['run_mode'] == 'distributed':
            assert num_gpus > 1, 'distributed mode can not be run with {} GPU'.format(colors.red(num_gpus))
            # NOTE: SyncBatchNorm may cause the inconsistance between non-distributed model and distributed model
            # i.e., 使用SyncBatchNorm，而后续代码中有数据(除Batch维)第一维数据交换，则可能导致SyncBatchNorm与BatchNorm结果不一致
            # see followings to find the reason
            # https://github.com/pytorch/pytorch/issues/62967#issuecomment-895695867 OR
            # https://github.com/fangwei123456/spikingjelly/issues/393
            self.log(logger.warning, colors.yellow('converting nn.BatchNorm to nn.SyncBatchNorm '
                                                   'may cause incompatibility if there exists channel-axis transpose in'
                                                   ' your model'))
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.to(dist.get_rank())
            model = parallel.DistributedDataParallel(model, device_ids=[dist.get_rank()])
            # if self.logable():
            self.log(logger.debug, 'run model in {} mode'.format(colors.red('distributed')))
        else:
            if num_gpus > 1 and self._config['run_mode'] == 'data-parallel':
                model = DataParallel(model, device_ids=range(num_gpus))
                model.cuda()
                # if self.logable():
                self.log(logger.debug, 'run model in {} mode'.format(colors.red('data-parallel')))
            else:
                device = torch.device('cuda:{}'.format(kwargs.get('device', 0)))
                model = model.to(device)
                # if self.logable():
                self.log(logger.debug, 'run model in {} mode with {} GPUs'.format(colors.red('single-gpu'),
                                                                                  colors.red(num_gpus)))
        return model

    def build_losses(self, loss_list: typing.List[str] | str) -> typing.Dict[str, typing.Callable]:
        """
        Builds a dictionary of loss functions based on the given list of loss names.

        Parameters:
            loss_list (List[str]): A list of names of loss functions.

        Returns:
            dict[str, Callable]: A dictionary mapping the names of loss functions to their corresponding functions.
        """
        if isinstance(loss_list, str):
            loss_list = [loss_list]
        from sigma.engine.metrics import losses
        # if self.logable():
        self.log(logger.info, 'building losses: {}'.format(colors.blue(loss_list)))
        return losses.get(loss_list)

    def build_optimizer(self, model: SigmaModule = None, **kwargs) -> optim.Optimizer:
        """
        Build the optimizer.

        Args:
            model: The model to be optimized.
            **kwargs: Additional keyword arguments.

        Returns:
            optim.Optimizer: The built optimizer.
        """
        from sigma.engine import optimizers
        # if self.logable():
        self.log(logger.info,
                 'building optimizer: {} with {}'.format(colors.blue(kwargs['name']),
                                                         ','.join([f'{colors.green(k)}[{colors.blue(v)}]' for k, v in kwargs['params'].items()])))
        if model is None:
            model = self._model
        return optimizers.get(kwargs['name'], model, **kwargs['params'])

    def build_schedulers(self, **kwargs):
        """
        Builds and returns a dictionary of schedulers based on the provided keyword arguments.

        Args:
            **kwargs: Keyword arguments where the keys represent the scheduler names and the values represent
                      either a single scheduler name or a list/tuple of scheduler names.

        Returns:
            dict: A dictionary where the keys are the scheduler names and the values are lists of scheduler
                  objects obtained from the 'schedulers' module.
        """
        from sigma.engine import schedulers
        assert hasattr(self, '_optimizer')
        s = {}
        for k, vs in kwargs.items():
            if vs is not None:
                # if self.logable():
                self.log(logger.info, 'building {} schedulers: {}'.format(colors.blue(k), colors.blue(vs)))
                if not isinstance(vs, (list, tuple)):
                    vs = [vs]
                s.update({k: [schedulers.get(v, self._optimizer) for v in vs]})
        return s

    def build_metrics(self, metric_list: typing.List[str] | str) -> typing.Dict[str, typing.Callable]:
        """
        Builds and returns a dictionary of metrics based on the given list of metric names.

        Parameters:
            metric_list (List[str]): A list of metric names to be included in the dictionary.

        Returns:
            Dict[str, Callable]: A dictionary where the keys are the metric names and the values are callable functions.
        """
        if isinstance(metric_list, str):
            metric_list = [metric_list]
        from sigma.engine.metrics import metrics
        # if self.logable():
        self.log(logger.info, 'building metrics: {}'.format(colors.blue(metric_list)))
        return metrics.get(metric_list)

    def load_checkpoint(self,
                        path: str,
                        model: SigmaModule = None,
                        strict_load: bool = True,
                        resume: bool = False,
                        exclude_parameter_pattern: str = None) -> None:
        """
        Loads a checkpoint from the given path.

        Args:
            model: The model to load the checkpoint into.
            path (str): The path to the checkpoint file.
            strict_load (bool, optional): Whether to strictly load the checkpoint. Defaults to True.
            resume (bool, optional): Whether to resume training from the checkpoint. Defaults to False.
            exclude_parameter_pattern:
        Returns:
            None
        """
        if model is None:
            assert hasattr(self, '_model'), 'no model is built. Please build model before loading checkpoint'
            model = self.model
        if path is None:
            # if self.logable():
            self.log(logger.info, colors.yellow('no checkpoint is given for loading'))
            return
        if not os.path.exists(path):
            # if self.logable():
            misc.log(logger.warning, 'checkpoint[{}] not exists: No such file or directory'.format(colors.red(path)))
            return
        tag = model.tag
        match = re.search(tag, path)
        if match is None:
            self.log(logger.warning,
                        colors.yellow(
                            'name of the weight file not match the model tag: {} <=> {}'.format(os.path.basename(path),
                                                                                                tag)))

        # if self.logable():
        self.log(logger.info, 'loading checkpoint from {} with strict[{}]'.format(colors.green(path),
                                                                                    colors.yellow(strict_load)))

        if torch.cuda.is_available():
            state = torch.load(path, map_location='cuda')
        else:
            state = torch.load(path, map_location='cpu')
        if 'weights' in state.keys():
            weights = state['weights']
        elif 'state_dict' in state.keys():
            weights = state['state_dict']
        else:
            weights = state

        # if module in key (DDP model), remove it
        if not hasattr(model, 'module'):
            # remove 'module.' if necessary
            state_dict = {k[7:] if 'module' in k else k: v for k, v in weights.items()}
        else:
            # add `module.` if necessary
            state_dict = {f'module.{k}' if 'module' not in k else k: v for k, v in weights.items()}

        if exclude_parameter_pattern is not None:
            assert strict_load is False, f'strict_load cannot be True given exclude_parameter_pattern[{exclude_parameter_pattern}]'
            state_dict = {k:v for k, v in state_dict.items() if re.search(exclude_parameter_pattern, k) is not None}
        model.load_state_dict(state_dict, strict=strict_load)

        if resume:
            if hasattr(self, 'optimizer') and 'optimizer' in state.keys():
                assert isinstance(self.optimizer, optim.Optimizer), \
                    f'optimizer must be an instance of `nn.Module` or its subclasses, given: {type(self.optimizer)}'
                self.optimizer.load_state_dict(state['optimizer'])
                self.log(logger.info, '{} resumed from {}'.format(colors.blue('optimizer'), colors.green(path)))

            if hasattr(self, 'epoch') and 'epoch' in state.keys():
                self._start_epoch = state['epoch']
                self.log(logger.info, '{} resumed from {}={}'.format(colors.blue('epoch'),
                                                                        colors.green(path),
                                                                        self._start_epoch))

            if hasattr(self, self._min_key) and self._min_key in state.keys():
                self._min_key_value = state[self._min_key]
                self.log(logger.info, '{} resumed from {}={}'.format(colors.blue(self._min_key),
                                                                        colors.green(path),
                                                                        self._min_key_value))

            if hasattr(self, 'iters') and 'iters' in state.keys():
                self._iters = state['iters']
                self.log(logger.info, '{} resumed from {}={}'.format(colors.blue('iters'),
                                                                        colors.green(path),
                                                                        self._iters))

            if hasattr(self, 'scaler') and 'scaler' in state.keys():
                self._scaler.load_state_dict(state['scaler'])
                self.log(logger.info, '{} resumed from {}'.format(colors.blue('scaler'), colors.green(path)))

    def save_checkpoint(self,
                        filename: str,
                        model: SigmaModule = None,
                        epoch: int = None,
                        min_key_value: float | str = None,
                        weights: dict = None) -> None:
        """
        Save a checkpoint of the model.

        Parameters:
            filename (str): The name of the file to save the checkpoint to.
            model (nn.Module, optional): The model to save. Default is None.
            epoch (int, optional): The current epoch. Default is None.
            min_key_value (float | str, optional): The min_key_value value. Default is None.
            weights (dict, optional): The model weights. Default is None.

        Returns:
            None: This function does not return anything.
        """
        if model is None:
            assert hasattr(self, '_model')
            model = self.model
        if weights is None:
            weights = model.state_dict()

        weights = {k[7:] if 'module' in k else k: v for k, v in weights.items()}
        state_dict = {'weights': weights,
                      'iters': self._iters}
        if epoch is not None:
            state_dict.update({'epoch': epoch})
        if min_key_value is not None:
            state_dict.update({self._min_key: min_key_value})
        if hasattr(self, 'scaler'):
            state_dict.update({'scaler': self._scaler.state_dict()})
        if hasattr(self, 'optimizer'):
            state_dict.update({'optimizer': self.optimizer.state_dict()})
        full_name = os.path.join(self._config['workspace'], 'ckpts',
                                 '{}-{}-{}'.format(model.tag,
                                                   utils.timestamp().replace('-', '_'),
                                                   filename))
        # if self.logable():
        self.log(logger.info, 'saving checkpoint to {}'.format(colors.green(full_name)))
        torch.save(state_dict, full_name)

    @property
    def lr(self) -> typing.List[float]:
        """
        Get a list of learning rates for all parameter groups in the optimizer.

        Returns:
            List[float]: A list of learning rates.
        """
        return [param_group['lr'] for param_group in self._optimizer.param_groups]

    def prepare_input(self,
                      data: typing.Dict[str, typing.Union[torch.Tensor, typing.Dict[str, torch.Tensor]]],
                      epoch: int,
                      it: int) -> None:
        self.model.preprocess(data)

    @staticmethod
    def save_pointcloud(path: str,
                        pred_labels: np.ndarray,
                        intrinsics: np.ndarray,
                        idx: int,
                        true_labels: torch.Tensor = None,
                        tof_labels: torch.Tensor = None,
                        images: np.ndarray = None):
        batch = pred_labels.shape[0]
        irs = [None] * batch if images is None else images

        for i, pred_label in enumerate(pred_labels.squeeze(axis=1)):
            if batch == 1:
                pcio.save_depth_as_pointcloud(os.path.join(path, '{}.ply'.format(idx)), pred_label, irs[i],
                                              intrinsics=intrinsics)
            else:
                os.makedirs(os.path.join(path, '{}'.format(idx)), exist_ok=True)
                pcio.save_depth_as_pointcloud(os.path.join(path, '{}/{}.ply'.format(idx, i)), pred_label, irs[i],
                                              intrinsics=intrinsics)

        if true_labels is not None:
            true_labels = true_labels.cpu().numpy().squeeze(axis=1)
            for i, true_label in enumerate(true_labels):
                if batch == 1:
                    pcio.save_depth_as_pointcloud(os.path.join(path, '{}_true.ply'.format(idx)), true_label, irs[i],
                                                  intrinsics=intrinsics)
                else:
                    pcio.save_depth_as_pointcloud(os.path.join(path, '{}/{}_true.ply'.format(idx, i)), true_label, irs[i],
                                                  intrinsics=intrinsics)
        if tof_labels is not None:
            tof_labels = tof_labels.numpy()
            for i, tof_label in enumerate(tof_labels):
                if batch == 1:
                    pcio.save_pointcloud(os.path.join(path, '{}_tof.ply'.format(idx)), tof_label)
                else:
                    pcio.save_pointcloud(os.path.join(path, '{}/{}_tof.ply'.format(idx, i)), tof_label)


    def evaluate(self, split: str = 'valid', max_iters: int = None, max_percentage: float = None) -> None:
        if split in self._datasets.keys():
            self.log(logger.info, 'evaluating on data split: {}'.format(colors.green(split)))
        else:
            self.log(logger.warning, 'invalid data split: {}. aborting evaluation'.format(colors.red(split)))
            return

        # use the minimal loss model
        if hasattr(self, 'champion'):
            # if self.logable():
            self.log(logger.info, 'loading champion state dict')
            state_dict = getattr(self, 'champion')
            if 'weights' in state_dict.keys():
                state_dict = state_dict['weights']
            self._model.load_state_dict(state_dict)

        os.makedirs(os.path.join(self._config['workspace'], 'prediction', split))
        path = os.path.join(self._config['workspace'], 'visualize', 'images',  split)
        os.makedirs(path, exist_ok=True)
        path_pc = os.path.join(self._config['workspace'], 'visualize',  'pointcloud', split)
        os.makedirs(path_pc, exist_ok=True)

        total_samples = len(self._datasets[split])

        if max_iters is None and max_percentage is None:
            max_iters = total_samples
        elif max_iters is None and max_percentage is not None:
            max_iters = int(total_samples * max_percentage)
        elif max_iters is not None and max_percentage is None:
            max_iters = min(max_iters, total_samples)
        else:
            max_iters = min(max_iters, int(total_samples * max_percentage))
        assert max_iters > 0, 'max_sample must be greater than 0, given {}'.format(max_iters)

        with misc.keep_status(self.model, False):
            with torch.inference_mode():
                if not isinstance(self._datasets[split], ProgressBar):
                    self._datasets[split] = ProgressBar(self._datasets[split])

                for idx, data in enumerate(self._datasets[split]):
                    if idx >= max_iters:
                        print('')
                        break

                    intrinsics = np.array([115.050003, 114.849998, 119.040001, 44.150002])
                    if 'intrinsics' in data.keys():
                        intrinsics = data['intrinsics'].cpu().numpy()

                    self.prepare_input(data, 0, idx)
                    self.valid_begin(data, 0, idx)
                    pred = self.model._infer(self.valid_step(data, 0, idx))

                    if 'mask' in data.keys():
                        mask = data['mask']
                        pred['pred'][~mask] = 0

                    np.savez(os.path.join(self._config['workspace'], 'prediction', split, '{}.npz'.format(idx)),
                             **{k:v.cpu().numpy for k,v in pred.items()})
                    pred_labels = pred['pred'].cpu().numpy()

                    batch_size = pred_labels.shape[0]
                    if 'true' in data.keys() and 'label' in data['true'].keys():
                        true_labels = data['true']['label'].cpu().numpy()

                        if batch_size > 1:
                            os.makedirs(os.path.join(path, '{}'.format(idx)), exist_ok=True)
                        for i, (pred_label, true_label) in enumerate(zip(pred_labels, true_labels)):
                            pred_label = pred_label.squeeze()
                            true_label = true_label.squeeze()
                            vis = np.concatenate([misc.visualize_map(pred_label, min_value=true_label.min(), max_value=true_label.max()),
                                                  misc.visualize_map(true_label)],
                                                 axis=1)
                            if batch_size > 1:
                                ios.save_others(os.path.join(path, '{}/{}.png'.format(idx, i)), vis)
                            else:
                                ios.save_others(os.path.join(path, '{}.png'.format(idx)), vis)
                    else:
                        if batch_size > 1:
                            os.makedirs(os.path.join(path, '{}'.format(idx)), exist_ok=True)
                        for i, pred_label in enumerate(pred_labels):
                            pred_label = pred_label.squeeze()
                            vis = misc.visualize_map(pred_label)
                            if batch_size > 1:
                                ios.save_others(os.path.join(path, '{}/{}.png'.format(idx, i)), vis)
                            else:
                                ios.save_others(os.path.join(path, '{}.png'.format(idx)), vis)

                    if 'image' in data.keys():
                        irs = data['image']
                    else:
                        irs = data['hist'].sum(dim=1).cpu().numpy()
                    Runner.save_pointcloud(path_pc, pred_labels, intrinsics, idx, data.get('true', {}).get('label', None), data.get('pointcloud', None), irs)

    def infer(self) -> typing.List[typing.Dict[str, float] | typing.Dict[str, typing.Dict[str, float]]]:
        """
        This function is used to perform inference.
        Parameters:
            None

        Returns:
            A list of dictionaries representing the inference results. Each dictionary can have two possible formats:
                - If the inference result is a single prediction, the dictionary will have keys representing the output
                    labels and values representing the corresponding probabilities.
                - If the inference result is a batch of predictions, the dictionary will have keys representing the
                    input samples and values representing dictionaries containing the output labels and their
                    corresponding probabilities.

        Example:
            [
                {
                    'label_1': 0.8,
                    'label_2': 0.2
                },
                {
                    'sample_1': {
                        'label_1': 0.7,
                        'label_2': 0.3
                    },
                    'sample_2': {
                        'label_1': 0.9,
                        'label_2': 0.1
                    }
                }
            ]
        """
        return self.evaluate(split='infer')

    def train(self) -> None:
        """
        Train the model.

        This function trains the model using the specified configuration. It performs the following steps:
        - Checks if the model is not None.
        - Sets the silent flag based on the run_mode and rank.
        - Initializes the epochs progress bar.
        - Sets the epochs and max_iters based on the configuration.
        - Updates the epochs and max_iters if they are None.
        - Resets the epoch and max_iters in the configuration.
        - Updates the datasets with the progress bar.
        - Retrieves the epoch_valid_freq and epoch_valid_thresh from the configuration.
        - Initializes the scaler if it is None.
        - Trains the model for each epoch.
        - Updates the epoch schedulers if they exist.
        - Validates the model if the epoch is valid.
        - Saves the checkpoint if the loss is the minimum.
        - Saves the champion checkpoint if training is stopped.
        - Evaluates the model on the 'valid' split.

        Parameters:
            None

        Returns:
            None
        """
        assert self.model is not None
        silent = self._config['run_mode'] == 'distributed' and dist.get_rank() != 0
        epochs = self._config['train']['epoch']
        max_iters = self._config['train']['max_iters']
        assert not (epochs is None and max_iters is None), 'train/epoch and train/max_iters cannot be both None'
        iter_per_epoch = max(len(self._datasets['train']), 1)
        if epochs is None:
            self._config['train']['epoch'] = math.ceil(max_iters / iter_per_epoch)
        elif max_iters is None:
            self._config['train']['max_iters'] = epochs * iter_per_epoch
        else:
            self._config['train']['max_iters'] = min(max_iters, epochs * iter_per_epoch)
            self._config['train']['epoch'] = math.ceil(self._config['train']['max_iters'] / iter_per_epoch)
        self.log(logger.info, 'reset epoch from {} to {}'.format(colors.blue(epochs if epochs is not None else 'None'),
                                                                 colors.blue(self._config['train']['epoch'])))
        self.log(logger.info,
                 'reset max_iters from {} to {}'.format(colors.blue(max_iters if max_iters is not None else 'None'),
                                                        colors.blue(self._config['train']['max_iters'])))
        epochs = ProgressBar(range(self._config['train']['epoch']),
                             start_index=self._start_epoch,
                             num_prompts=self._config.get('progressbar', {'num_prompts': 20})['num_prompts'],
                             keep_line=False,
                             silent=silent)
        cc = ['+', '=', '>']
        self._datasets.update(
            {k: epochs.sub(v, keep_line=False, cc=cc[idx]) for idx, (k, v) in enumerate(self._datasets.items())})

        epoch_valid_freq = self._config['train'].get('epoch_valid_freq', None)
        epoch_valid_thresh = self._config['train'].get('epoch_valid_thresh', 0)

        if self._scaler is None:
            self._scaler = amp.GradScaler(enabled=self._config['mixed_precision'])

        with misc.keep_status(self.model, set_train=True):
            for epoch in epochs:
                keep_training = self.train_epoch(epoch)
                if 'epoch_schedulers' in self._schedulers.keys():
                    [scheduler.step() for scheduler in self._schedulers['epoch_schedulers']]
                if (epoch_valid_freq is not None and epoch >= epoch_valid_thresh and (
                        epoch + 1) % epoch_valid_freq == 0) \
                        or (epoch + 1) == self._config['train']['epoch']:
                    if 'valid' in self._datasets.keys():
                        min_key_value = self.valid(epoch)[self._min_key]
                        if self.logable() and (self._min_key_value is None or min_key_value < self._min_key_value):
                            self._min_key_value = min_key_value
                            filename = f'epoch{epoch + 1}_iter{self._iters}_{self._min_key}{min_key_value}.pth'
                            self.save_checkpoint(filename, self.model, epoch, min_key_value)
                    else:
                        if self.logable():
                            filename = f'epoch{epoch + 1}_iter{self._iters}.pth'
                            self.save_checkpoint(filename, self.model, epoch, None)
                    if self._config['run_mode'] == 'distributed':
                        dist.barrier()
                if not keep_training:
                    state_dict = None
                    if hasattr(self, 'champion'):
                        # if self.logable():
                        self.log(logger.info, 'saving champion state dict')
                        state_dict = getattr(self, 'champion')
                    filename = f'epoch{epoch + 1}_iter{self._iters}_champion.pth'
                    self.save_checkpoint(filename, self.model, epoch, weights=state_dict)
                    if self._config['run_mode'] == 'distributed':
                        dist.barrier()
                    break
        # evaluate on the trained dataset
        evaluate_iters = self._config['train'].get('evaluate_iters', None)
        evaluate_percentage = self._config['train'].get('evaluate_percentage', None)
        if evaluate_iters is not None or evaluate_percentage is not None:
            self.evaluate(split='train', max_iters=evaluate_iters, max_percentage=evaluate_percentage)

        validate_iters = self._config['train'].get('validate_iters', None)
        validate_percentage = self._config['train'].get('validate_percentage', None)
        if validate_iters is not None or validate_percentage is not None:
            self.evaluate(split='valid', max_iters=validate_iters, max_percentage=validate_percentage)

        split = 'infer'  if 'infer' in self._datasets.keys() else 'valid'
        if split != 'valid' or (validate_iters is None and validate_percentage is None):
            self.evaluate(split=split)

    def train_epoch_begin(self, epoch: int) -> None:
        """
        Trains the model for the beginning of an epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            None
        """
        if self._config['run_mode'] == 'distributed':
            self._datasets['train'].items.sampler.set_epoch(epoch)
        self.model.train_epoch_begin()

    def train_epoch_step(self, epoch: int) -> typing.Tuple[typing.Dict[str, typing.List[float]], bool]:
        """
        Train the model for one epoch.

        Parameters:
            epoch (int): The current epoch number.

        Returns:
            Tuple[Dict[str, List[float]], bool]: A tuple containing two elements:
                - A dictionary where the keys are strings representing metrics and the values are lists of floats
                    representing the values of those metrics.
                - A boolean value indicating whether to continue training or not.
        """
        result = {}
        keep_training = True
        for it, data in enumerate(self._datasets['train']):
            res = self.train_iter(data, epoch, it)
            # [res.update({k, v} for k, v in )]
            for k, v in res.items():
                ''' append to dict result
                '''
                w = result.get(k, [])
                w.append(v)
                result.update({k: w})
            if self._config['train']['max_iters'] is not None and self._iters >= self._config['train']['max_iters']:
                keep_training = False
                break
        return result, keep_training

    def train_epoch_end(self,
                        result: typing.Dict[str, typing.List[float]],
                        epoch: int) -> None:
        """
        Train the model at the end of each epoch.

        Args:
            result (Dict[str, List[float]]): The result of the epoch.
            epoch (int): The current epoch number.

        Returns:
            None
        """
        self.model.train_epoch_end()

    def train_epoch(self, epoch: int) -> bool:
        """
        Trains a single epoch of the model.

        Args:
            epoch (int): The current epoch number.
        Returns:
            bool: A boolean value indicating whether to continue training or not.
        """
        self.train_epoch_begin(epoch)
        result, keep_training = self.train_epoch_step(epoch)  # cause different valid result even without training
        self.train_epoch_end(result, epoch)
        return keep_training

    def train_iter_begin(self,
                         data: typing.Dict[str, torch.Tensor],
                         epoch: int,
                         it: int) -> None:
        """
        Begin a training iteration.

        Args:
            data (Dict[str, torch.Tensor]): A dictionary containing the training data.
            epoch (int): The current epoch.
            it (int): The current iteration.

        Returns:
            None
        """
        self.model.train_iter_begin(data=data, epoch=epoch, it=it)

    def train_iter_step(self,
                        data: typing.Dict[str, torch.Tensor],
                        epoch: int,
                        it: int) -> typing.Union[typing.Dict[str, torch.Tensor],
                                                 typing.List[typing.Dict[str, torch.Tensor]]]:
        """
        Train a single iteration step.

        Args:
            data (Dict[str, torch.Tensor]): The input data for the iteration step.
            epoch (int): The current epoch number.
            it (int): The current iteration number.

        Returns:
            Dict[str, torch.Tensor]: The output data from the iteration step.
        """
        res = self._model(writer=self.writer, global_step=self._iters, **data)
        if self._teacher is None:
            return res
        with torch.inference_mode():
            teacher_output = self._teacher(writer=self.writer, global_step=self._iters, **data)
        return [res, teacher_output]

    def train_iter_end(self,
                       result: typing.Union[
                           typing.Dict[str, torch.Tensor], typing.List[typing.Dict[str, torch.Tensor]]],
                       data: typing.Dict[str, torch.Tensor],
                       epoch: int,
                       it: int) -> typing.Dict[str, float]:
        """
        Train the model for one iteration and return statistics.

        Parameters:
            result (typing.Dict[str, torch.Tensor]): The result of the model's forward pass.
            data (typing.Dict[str, torch.Tensor]): The input data for the model.
            epoch (int): The current epoch number.
            it (int): The current iteration number.

        Returns:
            typing.Dict[str, float]: A dictionary containing the statistics of the iteration.
        """
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self._config['mixed_precision']):
            # pred = result['pred']
            true = data['true']
            mask = data.get('mask', None)
            loss_dict = {k: loss(pred=result, true=true, mask=mask) for k, loss in self._losses.items()}
            loss = torch.sum(torch.stack(list(loss_dict.values())))

        self._scaler.scale(loss).backward()
        if self._config['train'].get('grad_norm_clip', None) is not None:
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), **self._config['train']['grad_norm_clip'])
        self._scaler.step(self._optimizer)

        if 'iters_schedulers' in self._schedulers.keys():
            [scheduler.step() for scheduler in self._schedulers['iters_schedulers']]

        self._scaler.update()

        self.model.train_iter_end()
        self._optimizer.zero_grad()

        # record CPU and GPU memory usage
        gpu_memory_total, gpu_memory_used, _ = misc.gpu_memory_info()
        iter_len = len(self._datasets['train'])
        # iter_size = utils.intsize(iter_len)
        spec = '[{}] [{}/{}] [{}/{}] lr:'
        message = spec.format(colors.blue(utils.timestamp(split=None)),
                               int(gpu_memory_used),
                               int(gpu_memory_total),
                               colors.blue(it + 1),
                               colors.blue(iter_len))
        lrs = self.lr
        for lr in lrs:
            message += '{},'.format(colors.green('{:.6f}'.format(lr)))

        loss = misc.sync_mean(loss, self._config['run_mode'] == 'distributed').item()
        loss_key = 'loss'
        if self._min_key == loss_key:
            loss_key = '{}'.format(styles.underline(loss_key))
        message += ' {}:{}'.format(loss_key, colors.green('{:.6f}'.format(loss)))

        stat = {'loss': loss}
        if len(loss_dict.keys()) > 1:
            for k, v in loss_dict.items():
                v = misc.sync_mean(v, self._config['run_mode'] == 'distributed').item()
                stat.update({k: v})
                if k == self._min_key:
                    k = '{}'.format(styles.underline(k))
                message += ', {}:{}'.format(k, colors.green('{:.6f}'.format(v)))
        res = result[0] if isinstance(result, list) else result
        pred_infer = self.model._infer(res)
        for k, metric in self._metrics.items():
            m = misc.sync_mean(metric(pred=pred_infer, true=true, mask=mask),
                               self._config['run_mode'] == 'distributed').item()
            stat.update({k: m})
            if k == self._min_key:
                k = '{}'.format(styles.underline(k))
            message += ', {}:{}'.format(k, colors.green('{:6f}'.format(m)))

        self.log_message(self._datasets['train'], message, header='T')
        if hasattr(self, '_writer') and self.logable():
            self.model.summary(self.writer, 'train', result, data, stat, self._iters, epoch, it)

        if self._config['run_mode'] == 'distributed':
            dist.barrier()

        self._iters += 1

        iters_valid_freq = self._config['train'].get('iters_valid_freq', None)
        iters_valid_thresh = self._config['train'].get('iters_valid_thresh', 0)
        if iters_valid_freq is not None and self._iters >= iters_valid_thresh and self._iters % iters_valid_freq == 0:
            if 'valid' in self._datasets:
                min_key_value = self.valid(epoch)[self._min_key]
                if self._min_key_value is None or min_key_value < self._min_key_value:
                    self._min_key_value = min_key_value
                    setattr(self, 'champion', self.model.state_dict())
                    if self.logable():
                        filename = f'epoch{epoch + 1}_iter{self._iters}_{self._min_key}{min_key_value}.pth'
                        self.save_checkpoint(filename, self.model, epoch, min_key_value)
            else:
                if self.logable():
                    filename = f'epoch{epoch + 1}_iter{self._iters}.pth'
                    self.save_checkpoint(filename, self.model, epoch, None)
            if self._config['run_mode'] == 'distributed':
                dist.barrier()
        return stat

    def train_iter(self,
                   data: typing.Dict[str, torch.Tensor],
                   epoch: int,
                   it: int) -> typing.Dict[str, float]:
        """
        Trains the model for one iteration.

        Args:
            data (Dict[str, torch.Tensor]): The input data for the iteration.
            epoch (int): The current epoch number.
            it (int): The current iteration number.

        Returns:
            Dict[str, float]: The result of the iteration.
        """
        self.prepare_input(data, epoch, it)
        self.train_iter_begin(data, epoch, it)
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self._config['mixed_precision']):
            res = self.train_iter_step(data, epoch, it)
        return self.train_iter_end(res, data, epoch, it)

    def valid(self, epoch: int) -> dict[str | typing.Any, typing.Any]:
        """
        Calculate the average loss for the validation dataset.

        Parameters:
            epoch (int): The current epoch.

        Returns:
            float: The average loss for the validation dataset.
        """
        with misc.keep_status(self.model, False):
            with torch.inference_mode():
                total_stat = {}
                if self._config['run_mode'] == 'distributed':
                    self._datasets['valid'].items.sampler.set_epoch(epoch)
                for idx, data in enumerate(self._datasets['valid']):
                    self.prepare_input(data, epoch, idx)
                    self.valid_begin(data, epoch, idx)
                    res = self.valid_step(data, epoch, idx)
                    self.valid_end(res, data, total_stat, epoch, idx)
            length = len(self._datasets['valid'])
            total_stat.update({k: v / length for k, v in total_stat.items()})
        return total_stat

    def valid_begin(self,
                    data: typing.Dict[str, torch.Tensor],
                    epoch: int,
                    idx: int) -> None:
        """
        A function to indicate the beginning of a valid iteration.

        Parameters:
            self (object): The object instance.
            data (Dict[str, torch.Tensor]): A dictionary containing the data.
            epoch (int): The current epoch number.
            idx (int): The index of the current iteration.

        Returns:
            None: This function does not return anything.
        """
        pass

    def valid_step(self,
                   data: typing.Dict[str, torch.Tensor],
                   epoch: int,
                   idx: int) -> typing.Dict[str, torch.Tensor]:
        """
        Validates a step in the training process.

        Args:
            data (typing.Dict[str, torch.Tensor]): A dictionary containing the input data.
            epoch (int): The current epoch number.
            idx (int): The index of the current step.

        Returns:
            typing.Dict[str, torch.Tensor]: A dictionary containing the output of the model.

        Raises:
            None
        """
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self._config['mixed_precision']):
            result = self._model(writer=self.writer, global_step=self._iters, **data)
        return result

    def valid_end(self,
                  result: typing.Dict[str, torch.Tensor],
                  data: typing.Dict[str, torch.Tensor],
                  total_stat: dict[str | typing.Any, typing.Any],
                  epoch: int,
                  idx: int) -> dict[str | typing.Any, typing.Any]:
        """
        Calculates the loss and metrics for the validation step of the training process.
        Args:
            result (typing.Dict[str, torch.Tensor]): The result dictionary containing the predicted values.
            data (typing.Dict[str, torch.Tensor]): The data dictionary containing the true values.
            total_stat (dict[str | Any, Any]): The total statistics dictionary.
            epoch (int): The current epoch number.
            idx (int): The index of the current step.
        Returns:
            dict[str | Any, Any]: A dictionary containing the calculated loss and metrics.
        Raises:
            None
        """
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self._config['mixed_precision']):
            # pred = result['pred']
            true = data['true']
            mask = data.get('mask', None)
            loss_dict = {k: misc.sync_mean(loss(pred=result, true=true, mask=mask),
                                           self._config['run_mode'] == 'distributed')
                         for k, loss in self._losses.items()}
            loss = torch.sum(torch.stack(list(loss_dict.values()))).item()
        stat = {'loss': loss}
        total_stat.update({'loss': total_stat.get('loss', 0) + loss})
        loss_key = 'loss'
        if self._min_key == loss_key:
            loss_key = '{}'.format(styles.underline(loss_key))
        message = '[{}] {}:{}'.format(colors.blue(utils.timestamp(split=None)),
                                      loss_key,
                                      colors.green('{:.6f}'.format(total_stat['loss'] / (idx + 1))))
        if len(loss_dict.keys()) > 1:
            for k, v in loss_dict.items():
                v = misc.sync_mean(v, self._config['run_mode'] == 'distributed').item()
                total_stat.update({k: total_stat.get(k, 0) + v})
                if k == self._min_key:
                    k = '{}'.format(styles.underline(k))
                message += ', {}:{}'.format(k, colors.green('{:.6f}'.format(total_stat[k] / (idx + 1))))
                stat.update({k: v})

        pred_infer = self.model._infer(result)
        for k, metric in self._metrics.items():
            m = misc.sync_mean(metric(pred=pred_infer, true=true, mask=mask),
                               self._config['run_mode'] == 'distributed').item()
            total_stat.update({k: total_stat.get(k, 0) + m})
            value = total_stat[k] / (idx + 1)
            if k == self._min_key:
                k = '{}'.format(styles.underline(k))
            message += ', {}:{}'.format(k, colors.green('{:6f}'.format(value)))
            stat.update({k: m})
        if hasattr(self, '_writer') and self.logable():
            self.model.summary(self.writer, 'valid', result, data, stat, self._iters, epoch, idx)
        self.log_message(self._datasets['valid'], message, header='V')
        return stat

    def dry_run(self) -> None:
        assert self._model is not None
        silent = self._config['run_mode'] == 'distributed' and dist.get_rank() != 0
        epochs = self._config['train']['epoch']
        max_iters = self._config['train']['max_iters']
        assert not (epochs is None and max_iters is None), 'train/epoch and train/max_iters cannot be both None'
        iter_per_epoch = len(self._datasets['train'])
        if epochs is None:
            self._config['train']['epoch'] = math.ceil(max_iters / iter_per_epoch)
        elif max_iters is None:
            self._config['train']['max_iters'] = epochs * iter_per_epoch
        else:
            self._config['train']['max_iters'] = min(max_iters, epochs * iter_per_epoch)
            self._config['train']['epoch'] = math.ceil(self._config['train']['max_iters'] / iter_per_epoch)
        self.log(logger.info, 'reset epoch from {} to {}'.format(colors.blue(epochs if epochs is not None else 'None'),
                                                                 colors.blue(self._config['train']['epoch'])))
        self.log(logger.info,
                 'reset max_iters from {} to {}'.format(colors.blue(max_iters if max_iters is not None else 'None'),
                                                        colors.blue(self._config['train']['max_iters'])))
        epochs = ProgressBar(range(self._config['train']['epoch']),
                             start_index=self._start_epoch,
                             num_prompts=self._config.get('progressbar', {'num_prompts': 100})['num_prompts'],
                             keep_line=False,
                             silent=silent)
        cc = ['+', '=', '>']
        self._datasets.update(
            {k: epochs.sub(v, keep_line=False, cc=cc[idx]) for idx, (k, v) in enumerate(self._datasets.items())})

        if self._scaler is None:
            self._scaler = amp.GradScaler(enabled=self._config['mixed_precision'])

        self.train_epoch_begin(0)
        for epoch in epochs:
            for it, data in enumerate(self._datasets['train']):
                self.prepare_input(data, epoch, it)
                self.train_iter_begin(data, epoch, it)
                message = '[{}] [{}/{}]'.format(colors.green(utils.timestamp(split=None)),
                                                colors.blue(it + 1),
                                                colors.blue(len(self._datasets['train'])))
                # self._datasets['train'].set_message(message)
                self.log_message(self._datasets['train'], message, header='T')
            self.model.train_epoch_end()
            break

    def logable(self, rank: int = 0) -> bool:
        """
        Check if the instance is loggable based on the run mode and rank.

        Args:
            rank (int, optional): The rank of the instance. Defaults to 0.

        Returns:
            bool: True if the instance is loggable, False otherwise.
        """
        if self._config['run_mode'] != 'distributed' or dist.get_rank() == rank:
            return True
        return False

    def log(self, fun: typing.Callable, message: str) -> None:
        """
        Logs a message using the provided function.

        Args:
            fun (typing.Callable): The function used to log the message.
            message (str): The message to be logged.

        Returns:
            None: This function does not return anything.
        """
        if self.logable():
            fun(message)
        self._logfile.write('{}\n'.format(fontstylebase.clear(message)))

    def log_message(self, progressbar: ProgressBar, message: str, header: str = None):
        progressbar.set_message(message)
        if header is None:
            self._logfile.write('{}\n'.format(message))
        else:
            self._logfile.write('<{}> {}\n'.format(header, fontstylebase.clear(message)))
        self._logfile.flush()
