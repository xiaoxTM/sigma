import logging
import torch
import sigma
import sigma.version as svn
from sigma.fontstyles import colors
from sigma import metrics as met
from sigma.utils import dict2str
from sigma.reporters import Logger, save_configs
from .checkpoint import CheckPointer
from .utils import set_seed
from . import optimizers, schedulers
import argparse
import os.path
import numpy as np
import shutil
import traceback


class Trainer():
    def __init__(self,
                 model,
                 optimizer='sgd',
                 scheduler=None,
                 lr=0.1,
                 epochs=100,
                 valid_percentage=0.2,
                 logger='run',
                 epochs_per_checkpoint=10,
                 min_epoch_for_checkpoint=0,
                 exp_path=None,
                 exp_name=None,
                 gpus=None,
                 openmp_num_threads=None,
                 **kwargs):
        self._model = model
        self._valid_percentage = valid_percentage
        self._lr = lr
        self._epochs = epochs
        self._num_samples = 0
        self._train_step = 0
        self._valid_step = 0
        self._total_loss = 0

        self._batch_size = kwargs.pop('batch_size')
        self._num_workers = kwargs.pop('num_workers')
        seed = kwargs.pop('seed')
        self._pin_memory = kwargs.pop('pin_memory')
        self._drop_last = not kwargs.pop('keep_last')

        if gpus is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        if openmp_num_threads is not None:
            if openmp_num_threads <= 0:
                openmp_num_threads = num_workers
            os.environ['OMP_NUM_THREADS'] = '{}'.format(openmp_num_threads)

        self._worker_fn = None
        if self._num_workers is not None and self._num_workers > 0:
            self._worker_fn = seed_workers(seed)

        self._logger = self.build_logger(os.path.join(exp_path, logger), 'w' if exp_name is None else 'a')
        self._checkpoint = self.build_checkpoint(os.path.join(exp_path,checkpoint))
        self._epochs_per_checkpoint = epochs_per_checkpoint
        self._min_epoch_for_checkpoint = min_epoch_for_checkpoint

        if svn.ge(torch.__version__, '1.4.0'):
            self.get_lr = self._get_last_lr
        else:
            self.get_lr = self._get_lr

    def build_optimizers(self, optimizers, parameters, lr):
        logging.info('building optimizer')
        self._optimizer = optimizers.get(optimizer, lr, parameters)

    def build_schedulers(self, scheduler, optimizer):
        logging.info('building scheduler')
        self._scheduler = schedulers.get(scheduler, self._optimizer)

    def build_checkpoint(self, checkpoint):
        logging.info('building checkpoint')
        if checkpoint is None:
            self._checkpoint = None
        elif isinstance(checkpoint, str):
            if len(checkpoint.strip())==0:
                self._checkpoint = None
            self._checkpoint = CheckPointer(checkpoint) # path of checkpoint
        elif isinstance(checkpoint, CheckPointer):
            self._checkpoint = checkpoint
        else:
            raise TypeError('cannot convert type {} into CheckPointer'.format(colors.red(type(checkpoint))))

    def build_logger(self, logger, mode):
        logging.info('building logger')
        if isinstance(logger, str):
            self._logger = Logger(logger, mode=mode) # file name of logger
        elif isinstance(logger, Logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError('cannot convert type {} into Logger'.format(colors.red(type(logger))))

    @classmethod
    def argument_parser(cls):
        parser = argparse.ArgumentParser(allow_abbrev=False)
        ####################
        # For reproducibility, set:
        #     --drop-last=True
        #     --keep-last=False
        #     --deterministic

        # experiment configuration
        parser.add_argument('--exp-root',type=str,default=None,help='root of the experiment directory')
        parser.add_argument('--extend-params',action='store_true')
        parser.add_argument('--exp-name',type=str,default=None,help='name of the experiment. if not None, resume states from checkpoint whoes name is exp-name')
        parser.add_argument('--not-save-model-file',action='store_true',default=False)
        # framework configuration
        parser.add_argument('--seed',type=int,default=1024)
        parser.add_argument('--check-dirty',action='store_true',default=False,help='switch to anomaly detection')
        parser.add_argument('--deterministic',action='store_true',default=False,help='switch to deterministic for cudnn')
        # train configuration
        parser.add_argument('--optimizer',type=str,default='sgd(momentum=0.9)')
        parser.add_argument('--scheduler',type=str,default=None)
        parser.add_argument('--lr',type=float,default=0.1)
        parser.add_argument('--epochs',type=int,default=200)
        parser.add_argument('--begin-epoch',type=int,default=None,help='begin epoch for train')
        parser.add_argument('--ipo',type=int,default=1,help='iterations per optimization')
        # dataset/dataloader
        parser.add_argument('--batch-size',type=int,default=10)
        parser.add_argument('--num-workers',type=int,default=0)
        parser.add_argument('--pin-memory',action='store_true',default=False) # according to Teacher Google, drop last batch that have small size will be more stable when training
        parser.add_argument('--keep-last',action='store_true',default=False)
        # openmp configuration
        parser.add_argument('--openmp-num-threads',type=int,default=None)
        # gpu configuration
        parser.add_argument('--gpus',type=str,default=None)
        # log configuration
        parser.add_argument('--checkpoint',type=str,default='checkpoints')
        parser.add_argument('--epochs-per-checkpoint',type=int,default=10)
        parser.add_argument('--min-epoch-for-checkpoint',type=int,default=0)
        parser.add_argument('--logger',type=str,default='run')
        parser.add_argument('--filename',type=str,default=None,help='confusion matrix filename for mode=`test` or features filename for mode=`retrieve`')
        # tensorboard configuration
        parser.add_argument('--tensorboard-log-dir',type=str,default=None)
        # visdom configuration
        parser.add_argument('--visdom-server',type=str,default=None)
        parser.add_argument('--visdom-port',type=int,default=8097)
        parser.add_argument('--visdom-base-url',type=str,default='/')
        parser.add_argument('--visdom-env',type=str,default='main')
        parser.add_argument('--visdom-log-to-filename',type=str,default=None)
        parser.add_argument('--visdom-offline',action='store_true',default=False)
        # progress bar configuration
        parser.add_argument('--progress-num-prompts',type=int,default=20)
        parser.add_argument('--progress-keep-line',action='store_true',default=False)
        parser.add_argument('--progress-silent',action='store_true',default=False)
        parser.add_argument('--progress-spec',type=str,default=None)
        parser.add_argument('--progress-nc',type=str,default='x+')
        parser.add_argument('--progress-cc',type=str,default='><')
        return parser

    def _get_last_lr(self):
        return self._scheduler.get_last_lr()

    def _get_lr(self):
        return [group['lr'] for group in self._optimizer.param_groups]

    def run_batch(self, x, labels, epoch=0, iteration=0):
        output = self._model(x, epoch, iteration)
        loss = self._model.loss(output, labels, epoch, iteration)
        return loss, self._model.predict(output, epoch, iteration), output

    def hitmap(self):
        preds, trues = self._model.prepare_metric()
        return met.hitmap(preds, trues, self._data_provider._num_classes)

    def run(self,train_set,test_set,
            valid_set=None,
            begin_epochs=0,
            ipo=1,
            weights=None,
            filename=None,
            parallel=None,
            **kwargs):
        if weights is not None:
            if not os.path.exist(weights):
                raise FileNotFoundError(colors.red(weights))
            self._model.load_state_dict(torch.load(weights))
        if devices is not None:
            if not isinstance(devices, (list, tuple)):
                devices = [devices]
            if parallel is None:
                self._model = self._model.to(devices[0])
            else:
                if  parallel == 'data_parallel':
                    self._model = torch.nn.DataParallel(self._model,devices)
                else:
                    raise ValueError(colors.red('unknown parallel mode'))
        if train_set is not None:
            self.train(train_set,valid_set,begin_epochs,ipo,**kwargs)
        if test_set is not None:
            self.valid(test_set,filename,**kwargs)

    def build_dataloader(self, train_set, valid_set):
        _train_set = train_set
        _valid_set = valid_set
        if valid_set is None:
            total_length = len(train_set)
            valid_length = int(train_length * self._valid_percentage)
            _train_set, _valid_set = random_split(train_set, [total_length-valid_length, valid_length])
        train_loader = DataLoader(_train_set,
                                  num_workers=self._num_workers,
                                  pin_memory=self._pin_memory,
                                  batch_size=self._batch_size,
                                  shuffle=True,
                                  drop_last=self._drop_last,
                                  worker_init_fn=self._worker_fn)
        valid_loader = DataLoader(_valid_set,
                                  num_workers=self._num_workers,
                                  pin_memory=self._pin_memory,
                                  batch_size=self._batch_size,
                                  shuffle=True,
                                  drop_last=self._drop_last,
                                  worker_init_fn=self._worker_fn)
        return train_loader, valid_loader

    def train(self, train_set, valid_set=None, begin_epoch=0, ipo=1,**kwargs):
        ebar = sigma.ProgressBar(range(begin_epoch, self._epochs), **kwargs)
        self._train_step = 0
        self._valid_step = 0
        train_data = []
        train_labels = []
        train_loader, valid_loader = None, None
        for epoch in ebar:
            if train_loader is None or valid_set is None:
                train_loader, valid_loader = self.build_dataloader(train_set, valid_set)
            total_iters = len(train_loader)
            ibar = ebar.sub(enumerate(train_loader))
            ####################
            # Train
            ####################
            self.mode(train=True)
            self.on_epoch_begin(epoch, begin_epoch, ebar)
            for idx, (data, label) in ibar:
                self._train_step += 1
                if (idx+1) % ipo == 0 or (idx+1) == total_iters:
                    self.on_batch_begin(epoch, idx, ebar)
                    self.train_on_batch(train_data, train_labels, epoch, idx)
                    self.on_batch_end(epoch, idx, ebar)
                    train_data = []
                    train_labels = []
                train_labels.append(train_label)
                train_data.append(train_datum)
            self.on_epoch_end(epoch, begin_epoch, ebar)

            vbar = ebar.sub(enumerate(valid_loader))
            ####################
            # valid
            ####################
            with torch.no_grad():
                self.mode(train=False)
                self.on_epoch_begin(epoch, begin_epoch, ebar)
                for idx, (data, label) in vbar:
                    self._valid_step += 1
                    valid_data, valid_label = self.prepare_data(data, label, device)
                    self.on_batch_begin(epoch, idx, ebar)
                    self.valid_on_batch(valid_data, valid_label, epoch, idx)
                    self.on_batch_end(epoch, idx, ebar)
                self.on_epoch_end(epoch, begin_epoch, ebar)
                # clean up internal values, e.g., total_loss, num_samples, model.preds, model.trues
                if self._checkpoint is not None:
                    if self._model.validate_metric(): 
                        # save better performance weight, does not create latest weight link
                        self._checkpoint.save_best(self._model, self._optimizer, self._scheduler, epoch)
                    if self._min_epoch_for_checkpoint <= epoch and (epoch+1) % self._epochs_per_checkpoint == 0:
                        self._checkpoint.save(self._model, self._optimizer, self._scheduler, epoch)

    def valid(self, dataset, filename=None, **kwargs):
        valid_loader = DataLoader(dataset,
                                  num_workers=self._num_workers,
                                  pin_memory=self._pin_memory,
                                  batch_size=self._batch_size,
                                  shuffle=True,
                                  drop_last=self._drop_last,
                                  worker_init_fn=self._worker_fn)
        bar = sigma.ProgressBar(range(1), **kwargs)
        vbar = bar.sub(enumerate(valid_loader))
        with torch.no_grad():
            self.mode(False)
            for epoch in bar:
                self.on_epoch_begin(epoch, 0, bar)
                for idx, (data, label) in vbar:
                    valid_data, valid_label = self.prepare_data(data, label, device)
                    self.on_batch_begin(epoch, idx, bar)
                    self.valid_on_batch(valid_data, valid_label, 0, idx)
                    self.on_batch_end(epoch, idx, bar)
                if filename is not None:
                    self.confusion_matrix(filename, self._model._preds, self._model._trues)
                self.on_epoch_end(epoch, 0, bar)

    @property
    def loss(self):
        return self._total_loss / self._num_samples

    def train_on_batch(self, data, labels, epoch=0, iteration=0):
        def closure():
            self._optimizer.zero_grad()
            losses = 0
            for datum,label in zip(data, labels):
                train_datum, train_label = self.prepare_data(datum, label)
                loss, pred, _ = self.run_batch(datum, label, epoch, iteration)
                loss.backward()
                self._num_samples += datum.size(0)
                self._total_loss += (loss.item() * datum.size(0))
                self._model.metric(pred, label)
                losses += loss.item()
            return losses
        self._optimizer.step(closure)

    def valid_on_batch(self, data, labels, epoch=0, iteration=0):
        loss, preds, logits = self.run_batch(data, labels, epoch, iteration)
        self._num_samples += data.size(0)
        self._total_loss += (loss.item() * data.size(0))
        self._model.store(preds, labels)
        return logits

    def on_batch_begin(self, epoch, batch, bar):
        self._model.on_batch_begin(epoch, batch)

    def on_batch_end(self, epoch, batch, bar):
        if isinstance(self._scheduler, tuple(schedulers.__batch__)):
            self._scheduler.step()
        loss = self.loss
        message = 'loss:{:0<.6F}'.format(loss)
        cost, metrics = self._model.on_batch_end(epoch, batch)
        if cost is not None:
            message = '{} {}'.format(message, dict2str(cost))
        if metrics is not None:
            message = '{} {}'.format(message, dict2str(metrics))
        bar.set_message(message)

    def on_epoch_begin(self, epoch, begin_epoch, bar):
        self._model.on_epoch_begin(epoch)

    def on_epoch_end(self, epoch, begin_epoch, bar):
        loss = self.loss
        if not (self._model.training or self._scheduler is None) and \
           isinstance(self._scheduler, tuple(schedulers.__epoch__)):
            # step after validation/testing
            self._scheduler.step()
        phase = 'train' if self._model.training else 'valid'
        message = 'loss:{:0<.6F}'.format(loss)
        cost, metrics = self._model.on_epoch_end(epoch)
        if cost is not None:
            message = '{} {}'.format(message, dict2str(cost))
        if metrics is not None:
            message = '{} {}'.format(message, dict2str(metrics))
        bar.set_message(message)
        self.log(epoch, message)
        self.clean()

    def mode(self, train=True):
        if train:
            self._model.train()
            if self._logger is not None:
                self._logger.train()
        else:
            self._model.eval()
            if self._logger is not None:
                self._logger.eval()

    def resume(self, mode='train'):
        if self._checkpoint is None:
            return None
        assert isinstance(self._checkpoint, CheckPointer), 'cannot call resume from checkpoint by `{}`'.format(type(self._checkpoint))
        assert mode in ['train', 'trial', 'valid', 'test']
        if mode in ['train', 'trial']:
            ckpt = self._checkpoint.resume(self._model, self._optimizer, self._scheduler, mode='latest')
        elif mode in ['valid', 'test']:
            ckpt = self._checkpoint.resume(self._model, self._optimizer, self._scheduler, mode='best')
        return ckpt

    def log(self, epoch, message=None, iteration=None):
        if self._logger is not None:
            assert self._num_samples > 0,  'You have to `train/test` the model before log it'
            loss = self._total_loss / self._num_samples
            message = 'lr:{} {}'.format(self.optimizer.param_groups[0]['lr'], message)
            self._logger.log(message, epoch+1)

    def clean(self):
        self._num_samples = 0
        self._total_loss = 0
        self._model.clean()

    def prepare_data(self, data, label):
        return self._model.prepare_data(data, label)

    def mode(self, train=True):
        if train:
            self._model.train()
        else:
            self._model.eval()

    @property
    def optimizer(self):
        return self._optimizer

    def confusion_matrix(self, filename, preds, truth, *args, **kwargs):
        cm = metric.confusion_matrix(truth, preds, *args, **kwargs)
        np.savetxt(filename, cm, fmt='%d')
        return cm

    def graph(self, filename, x, params=None):
        if visualization:
            g = make_dot(self.forward(x), params=params)
            g.save(filename)
        else:
            logging.warning('Ignore visualization of network because torchviz not installed')

class Experiment(Trainer):
    def __init__(self,
                 model,
                 checkpoint='checkpoints',
                 tensorboard_log_dir=None,
                 visdom_server=None,
                 visdom_port=8097,
                 visdom_base_url='/',
                 visdom_env='main',
                 visdom_log_to_filename=None,
                 visdom_offline=False,
                 **kwargs):
        super(Experiment, self).__init__(model,
                                                **kwargs)
        self._writer = None
        self._webviz = None
        if tensorboard_log_dir is not None:
            try:
                import torch.utils.tensorboard.writer as ttw
                self._writer = ttw.SummaryWriter(tensorboard_log_dir)
            except Exception as e:
                logging.warning('tensorboard disabled.')
                logging.debug(traceback.print_exc())
        if visdom_server is not None:
            try:
                from sigma.reporters.webviz import Webviz
                self._webviz = Webviz(server=visdom_server,
                                      port=visdom_port,
                                      base_url=visdom_base_url,
                                      env=visdom_env,
                                      log_to_filename=visdom_log_to_filename,
                                      offline=visdom_offline)
            except Exception as e:
                self._webviz = None
                logging.warning('visdom disabled.')
                logging.debug(traceback.print_exc())

    def log_grad(self):
        if self._writer is not None:
            for n,p in self._model.named_parameters():
                if p.requires_grad:
                    self._writer.add_histogram('grad/{}'.format(n), p.grad, self._train_step)

    def train_on_batch(self, data, labels, epoch=0, iteration=0):
        def closure():
            self._optimizer.zero_grad()
            losses = 0
            for datum,label in zip(data,labels):
                train_datum,train_label = self.prepare_data(datum,label)
                loss, preds, logits = self.run_batch(datum, label, epoch, iteration)
                loss.backward()
                self.log_grad()
                self._num_samples += data.size(0)
                if self._webviz is not None:
                    key = 'train_loss_iter'
                    self._webviz.line(key, X=np.asarray([self._train_step]),
                                       Y=np.asarray([loss.item()]),
                                       name='total-loss',
                                       update='append',
                                       opts=dict(title='train loss iter',
                                                 xlabel='iter',
                                                 ylabel='loss'))
                self._total_loss += (loss.item() * data.size(0))
                self._model.metric(preds, label)
            return losses
        self._optimizer.step(closure)

    def valid_on_batch(self, data, labels, epoch=0, iteration=0):
        loss, preds, logits = self.run(data, labels, epoch, iteration)
        self._num_samples += data.size(0)
        self._total_loss += (loss.item() * data.size(0))
        if self._webviz is not None:
            key = 'valid_loss_iter'
            self._webviz.line(key,X=np.asarray([self._valid_step]),
                                  Y=np.asarray([loss.item()]),
                                  name='total-loss',
                                  update='append',
                                  opts=dict(title='valid loss iter',
                                            xlabel='iter',
                                            ylabel='loss'))
        self._model.store(preds, labels)
        return logits

    def on_batch_end(self, epoch, batch, bar):
        if isinstance(self._scheduler, tuple(schedulers.__batch__)):
            if self._webviz is not None:
                self._webviz.line('lr_iter',X=np.asarray([self._train_step]),
                                            Y=np.asarray([self.get_lr()]),
                                            name='lr-iter',
                                            update='append',
                                            opts=dict(title='iter learning rate',
                                                      xlabel='iter',
                                                      ylabel='learning rate'))
            self._scheduler.step()
        loss = self.loss
        message = 'loss:{:0<.6F}'.format(loss)
        step = self._train_step if self._model.training else self._valid_step
        cost, metrics = self._model.on_batch_end(epoch, batch)
        phase = 'train' if self._model.training else 'valid'
        if cost is not None:
            message = '{} {}'.format(message, dict2str(cost))
            if self._webviz is not None:
                key = '{}_loss_iter'.format(phase)
                for k,v in cost.items():
                    self._webviz.line(key,X=np.asarray([step]),
                                          Y=np.asarray([v]),
                                          name=k,
                                          update='append',
                                          opts=dict(showlegend=True))
        if metrics is not None:
            message = '{} {}'.format(message, dict2str(metrics))
            if self._webviz is not None:
                key = '{}_metric_iter'.format(phase)
                for k,v in metrics.items():
                    self._webviz.line(key,X=np.asarray([step]),
                                          Y=np.asarray([v]),
                                          name=k,
                                          update='append',
                                          opts=dict(xlabel='iter',
                                                    ylabel='metric',
                                                    title=key.replace('_', ' '),
                                                    showlegend=len(metrics)>1))
        if self._writer is not None:
            if self._model.training:
                self._writer.add_scalar('loss/train/total/iter', loss, self._train_step)
                for i, param in enumerate(self.optimizer.param_groups):
                    self._writer.add_scalar('lr/train/iter/{}'.format(i), param['lr'], self._train_step)
            else:
                self._writer.add_scalar('loss/valid/total/iter', loss, self._valid_step)
        bar.set_message(message)

    def on_epoch_end(self, epoch, begin_epoch, bar):
        if not (self._model.training or self._scheduler is None) and \
           isinstance(self._scheduler, tuple(schedulers.__epoch__)): # step after validation/testing
            if self._webviz is not None:
                self._webviz.line('lr_epoch',X=np.asarray([self._train_step]),
                                             Y=np.asarray([self.get_lr()]),
                                             name='lr-epoch',
                                             update='append',
                                             opts=dict(title='epoch learning rate',
                                                       xlabel='epoch',
                                                       ylabel='learning rate'))
            self._scheduler.step()
        loss = self.loss
        message = 'loss:{:0<.6F}'.format(loss)
        cost, metrics = self._model.on_epoch_end(epoch)
        phase = 'train' if self._model.training else 'valid'
        if self._webviz is not None:
            self._webviz.line('loss_epoch',X=np.asarray([epoch]),
                                           Y=np.asarray([loss]),
                                           name='{}-loss'.format(phase),
                                           update='append',
                                           opts=dict(title='loss epoch',
                                                     xlabel='epoch',
                                                     ylabel='loss',
                                                     showlegend=True))
            if self._model.training:
                hitmap = self.hitmap()
                self._webviz.heatmap('valid-hitmap',X=hitmap,
                                                    name='valid-hitmap',
                                                    update='replace',
                                                    opts=dict(title='confusion matrix',
                                                              xlabel='trues',
                                                              ylabel='preds'))
        if cost is not None:
            message = '{} {}'.format(message, dict2str(cost))
            if self._webviz is not None:
                for k,v in cost.items():
                    self._webviz.line('loss_epoch',
                                      X=np.asarray([epoch]),
                                      Y=np.asarray([v]),
                                      name='{}-{}'.format(phase,k),
                                      update='append',
                                      opts=dict(xlabel='epochs',
                                                ylabel='loss',
                                                showlegend=True))
        if metrics is not None:
            message = '{} {}'.format(message, dict2str(metrics))
            if self._webviz is not None:
                key = 'metric_epoch'
                for k,v in metrics.items():
                        self._webviz.line(key,X=np.asarray([epoch]),
                                              Y=np.asarray([v]),
                                              name='{}-{}'.format(phase,k),
                                              update='append',
                                              opts=dict(xlabel='epoch',
                                                        ylabel='metric',
                                                        title='metric epoch',
                                                        showlegend=len(metrics)>1))
        if self._writer is not None:
            self._writer.add_scalar('loss/{}/total/epoch'.format(phase), loss, epoch)
            if self._model.training:
                for i, param in enumerate(self.optimizer.param_groups):
                    self._writer.add_scalar('lr/epoch/{}'.format(i), param['lr'], epoch)
            self._writer.flush()
        bar.set_message(message)
        self.log(epoch, message)
        self.clean()


# def run(exp, begin_epoch=0, mode='train', test_after_train=False, filename=None, **kwargs):
#     parameters = {}
#     for k, v in kwargs.items():
#         if k.startswith('progress_'):
#             parameters.update({k.replace('progress_', ''):v})
#     if mode == 'train':
#         exp.train(begin_epoch, test_after_train=test_after_train,**parameters)
#     elif mode == 'trial':
#         exp.trial(begin_epoch, test_after_train=test_after_train,**parameters)
#     elif mode == 'test':
#         exp.test(filename, **parameters)
#     else:
#         raise NotImplementedError('operation {} not supported'.format(colors.red(mode)))


#def init_dir(expname, parent=None):
#    if expname is None:
#        expname = 'exp-{}'.format(met.timestamp())
#    if parent is not None:
#        expname = os.path.join(parent, expname)
#    os.makedirs(expname, exist_ok=True)
#    return expname


# def init_dir(args):
#     params = '{}-{}-{}-{}-{}-{}-{}-{}'.format(args.module,
#                                               args.weight_initializer,
#                                               args.bias_initializer,
#                                               args.optimizer,
#                                               args.scheduler,
#                                               args.loss,
#                                               args.lr,
#                                               args.batch_size)
#     if args.exp_name is None:
#         exp_path = 'exp-{}'.format(met.timestamp())
#         if args.extend_params:
#             exp_path = '{}-{}'.format(exp_path,params)
#     if args.exp_root is not None:
#         exp_path = os.path.join(args.exp_root,exp_path)
#     os.makedirs(exp_path,exist_ok=True)
#     return exp_path


# def init(Exp):
#     parser = Exp.argument_parser()
#     args, remain_args = parser.parse_known_args()
#     logging.info('{}-ing {} on {}'.format(colors.green(args.mode),
#                                           colors.red(args.model),
#                                           colors.blue(args.dataset)))
#     args.exp_path = init_dir(args)
#     logging.info('experimenting on {}'.format(colors.red(args.exp_path)))

#     args.cuda = torch.cuda.is_available()
#     if args.cuda:
#         devices = torch.cuda.device_count()
#         logging.info('using {} devices'.format(colors.red('{} GPUs'.format(devices))))
#     else:
#         logging.info('using CPU')

#     torch.autograd.set_detect_anomaly(args.check_dirty)
#     return args, remain_args


# def main(Exp):
#     import importlib
#     args, remain_args = init(Exp)
#     set_seed(args.seed, args.deterministic)

#     # prepare dataset
#     dataset_path = args.dataset.replace('/', '.')
#     DataProvider = importlib.import_module(dataset_path).DataProvider

#     # prepare for model
#     model_path = args.model.replace('/','.')
#     package = importlib.import_module(model_path)
#     Model = eval('package.{}'.format(args.module))
#     module_parser = Model.argument_parser()
#     margs = module_parser.parse_args(remain_args)
#     if not args.not_save_model_file:
#         model_name = model_path.rsplit('.',1)[-1]
#         model_filename = os.path.join(args.exp_path, model_name + '.py')
#         if os.path.exists(model_filename):
#             ans = input('file: {} already exists. overwrite?: '.format(colors.red(model_filename)))
#             if ans.lower() != 'y':
#                 print('Exit !!!')
#                 exit()
#         shutil.copyfile(model_path.replace('.','/')+'.py', model_filename)

#     dargs = sigma.arg2dict(args, excludes=['model','mode','begin_epoch','test_after_train','filename','module','not_save_model_file'])
#     # push necessary parameters for model building
#     dargs.update(vars(margs))

#     logging.info('loading data')
#     data_provider = DataProvider(**dargs)

#     dargs.update({'num_classes': data_provider.num_classes})
#     logging.info('building model')
#     model = Model(**dargs)

#     dargs.pop('num_classes')

#     # prepare experiment
#     logging.info('building experiment')
#     exp = Exp(model, data_provider, **dargs)

#     # assign `exp` to `model`
#     model._exp = exp

#     cuda = torch.cuda.is_available()
#     if cuda:
#         devices = torch.cuda.device_count()
#         if devices == 1:
#             exp._model.to(torch.device('cuda:0'))

#     # resume from checkpoint if is_available
#     begin_epoch=args.begin_epoch
#     if begin_epoch is None:
#         begin_epoch = 0
#     if args.exp_name is not None and os.path.exists(args.exp_path):
#         ckpt = exp.resume(mode=args.mode)
#         if args.begin_epoch is None or args.begin_epoch < 0:
#             begin_epoch = ckpt['begin-epoch'] + 1
#     elif args.mode in ['test', 'valid']:
#         raise ValueError('You must provide checkpoint to resume model from for mode: {}'.format(args.mode))

#     # training settings
#     save_configs(args.exp_path, margs, args)
#     run(exp, begin_epoch, args.mode, args.test_after_train, args.filename, **dargs)
