import torch
from torch.utils.tensorboard import SummaryWriter
import os.path
import numpy as np

from sigma.engine import misc
from sigma.engine.misc import pcio, ios

from sigma.core.reporters.progressbar import ProgressBar
from sigma.core.fontstyles import colors

import logging

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def infer_from_dir(args) -> None:
    from sigma.engine.misc import stream

    path = os.path.join(args['workspace'], 'visualize')
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(args['workspace'], 'prediction'), exist_ok=True)
    os.makedirs(os.path.join(path, 'pointcloud'), exist_ok=True)
    os.makedirs(os.path.join(path, 'images'), exist_ok=True)

    source = args['path']
    if source is None:
        source = args['infer']['path']
        root = args['infer'].get('root', None)
        if root is not None:
            source = os.path.join(root, source)

    model = misc.build_model(args)
    dataset = stream.Stream(source,
                            root=args['infer']['base'],
                            scale=args.get('scale', 0.00025),
                            augments=args.get('augments', []))

    pbar = ProgressBar(total=len(dataset), num_prompts=50)
    writer = None
    if args['train'].get('summary', False):
        logger.info(colors.blue('tensorboard summary enabled'))
        writer = SummaryWriter(os.path.join(args['workspace'], 'logs'))

    model.eval()
    with torch.no_grad():
        for step, data in zip(pbar, dataset):
            # NOTE: batch_size = 1
            true = {}
            mask = None
            hist = data['hist']
            name = data['name']
            if 'mask' in data.keys():
                mask = data['mask']
            if 'label' in data.keys():
                true_depth = data['label']
                true.update({'label': true_depth[np.newaxis]})
            if mask is None:
                h, w, _ = hist.shape
                mask = np.ones((h, w), dtype=np.uint8)
            samples = {'hist': hist, 'mask': mask}
            name = os.path.splitext(os.path.basename(name))[0]

            # [1, height, width]
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args['mixed_precision']):
                pred = model.infer(**misc.preprocess(samples), writer=writer, global_step=step)
                # save
                np.savez(os.path.join(args['workspace'], 'prediction', name),
                         **{k: v.cpu().numpy for k, v in pred.items()})
                # save as pointcloud
                depth = pred['pred'].cpu().numpy().squeeze()
                image = data.get('image', None)
                pcio.save_depth_as_pointcloud(os.path.join(path, 'pointcloud', name+'.ply'), depth, image=image, intrinsics=args['intrinsics'])
                if 'depth' in data.keys():
                    true_depth = data['depth'].squeeze()
                    pcio.save_depth_as_pointcloud(os.path.join(path, 'pointcloud', name+'_true.ply'), true_depth, image=image, intrinsics=args['intrinsics'])
                    depth_map = np.concatenate([misc.visualize_map(depth, min_value=true_depth.min(), max_value=true_depth.max()),
                                                misc.visualize_map(true_depth)], axis=1)
                else:
                    depth_map = misc.visualize_map(depth)
                ios.save_others(os.path.join(path, 'images', name+'.png'), depth_map)
            message = '[{}] {}'.format(colors.green('infer'), colors.blue(name))
            pbar.set_message(message)


def infer(args) -> None:
    if args['path'] is not None or args['infer']['path'] is not None:
        infer_from_dir(args)
        return
    from .runner import Runner
    if args['infer']['checkpoint'] is not None:
        if os.path.exists(args['infer']['checkpoint']):
            args['train']['checkpoint']['path'] = args['infer']['checkpoint']
        else:
            if args['train']['checkpoint']['path'] is None:
                logger.fatal('neither infer.checkpoint nor train.checkpoint.path is provided')
    else:
        if args['train']['checkpoint']['path'] is None:
            logger.fatal('neither infer.checkpoint nor train.checkpoint.path is provided')
    Runner(args).infer()
