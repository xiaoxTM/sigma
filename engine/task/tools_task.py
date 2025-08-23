import logging
import os.path
import numpy as np
import typing
import copy
import matplotlib.pyplot as plt
from sigma.engine import misc
from sigma.engine.misc import ios
from sigma.engine.metrics import metrics
from sigma.engine.dataloaders import augments as augs

import cv2
from sigma.core.reporters.progressbar import ProgressBar
from sigma.core.fontstyles import colors
from sigma.core.utils import config as cfg

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def analyze(args):
    from src.misc.metrics import Analyzer2
    augments = augs.get(args.get('augments', []))
    scales = args['scale']
    if args['baseline'] is not None and args['intrinsics'] is not None:
        analyzer = Analyzer2(args['baseline'],
                             args['intrinsics'] * args.get('intrinsic_scale', 1.0),
                             args['ranges'],
                             args['trim'],
                             args['filters'],
                             visualize=args['visualize'],
                             save_pointcloud=args['save_pointcloud'],
                             filelog=os.path.join(args['workspace'], 'file.log'),
                             path=args['workspace'])
        dataset = ios.load_dataset(args['predicted_disparity'], args['true_disparity'], args['images'], args['mask'])
        logger.info(f'total {len(dataset)} samples found')
        os.makedirs(os.path.join(args['workspace'], 'prediction'), exist_ok=True)
        os.makedirs(os.path.join(args['workspace'], 'pointcloud'), exist_ok=True)
        os.makedirs(os.path.join(args['workspace'], 'analyze'), exist_ok=True)
        dbar = ProgressBar(total=len(dataset), num_prompts=100)

        for _, data in zip(dbar, dataset):
            pred_name = data['pred']
            true_name = data['true']
            image_name = data['image']
            pred = ios.load_disp(pred_name, scales[1])
            mask_name = data.get('mask', None)
            true = ios.load_disp(true_name, scales[0])

            if mask_name is not None:
                mask = ios.load_others(mask_name, mode='L')
            else:
                mask = np.ones(true.shape, dtype=np.uint8)
            true_data = augments({'disp': true, 'mask': mask})
            mask = true_data['mask']
            score = np.load(pred_name).get('score', None)

            image = None
            if image_name is not None:
                image = ios.load_others(image_name)
                if args['stereo']:
                    image = np.split(image, 2, axis=1)[0]
                image = np.expand_dims(image, axis=0)
            if analyzer is not None:
                if pred.ndim == 2:
                    pred = np.expand_dims(pred, axis=(0, 1))
                elif pred.ndim == 3:
                    pred = np.expand_dims(pred, axis=0)
                inputs = {'disp': pred}
                if score is not None:
                    if score.ndim == 2:
                        score = np.expand_dims(score, axis=(0, 1))
                    elif score.ndim == 3:
                        score = np.expand_dims(score, axis=0)
                    inputs.update({'score': score})
                name = os.path.splitext(os.path.basename(pred_name))[0]
                true_disp = true_data['disp']
                if true_disp.ndim == 2:
                    true_disp = np.expand_dims(true_disp, axis=(0, 1))
                elif true_disp.ndim == 3:
                    true_disp = np.expand_dims(true_disp, axis=0)
                analyzer(inputs,
                         image,
                         mask > 0,
                         {'disp': true_disp},
                         name=name,
                         image_mean=args['image_mean'])
        analyzer.summary(os.path.join(args['workspace'], 'stat.txt'))
    else:
        dataset = ios.load_dataset(args['predicted_disparity'], args['true_disparity'], args['images'], args['mask'])
        logger.info(f'total {len(dataset)} samples found')
        dbar = ProgressBar(total=len(dataset), num_prompts=100)
        os.makedirs(os.path.join(args['workspace'], 'analyze'), exist_ok=True)
        for _, data in zip(dbar, dataset):
            pred_name = data['pred']
            image_name = data['image']
            pred = ios.load_disp(pred_name, scales[1]).squeeze()
            pred_map = misc.visualize_map(pred)
            if image_name is not None:
                image = ios.load_others(image_name)
                if image.shape[-2] != pred.shape[-1] or image.shape[-3] != pred.shape[-2]:
                    pred_map = cv2.resize(pred_map, (image.shape[-2] // 2, image.shape[-3]),
                                          interpolation=cv2.INTER_LINEAR)
                    ios.save_others(os.path.join(args['workspace'], 'analyze', os.path.basename(image_name)),
                                    np.concatenate([image, pred_map], axis=1))
                else:
                    ios.save_others(os.path.join(args['workspace'], 'analyze', os.path.basename(image_name)), pred_map)


def anaplot(args):
    font_size = args['font_size']
    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    models = os.listdir(args['path'])

    num_models = len(models)
    width = 0.8 / num_models
    cols = args.get('ncols', 1)
    rows = args.get('nrows', 1)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True)
    mean_stat = {}
    bars = []
    bar_label_dict = args['bar_label_dict']
    if bar_label_dict is None:
        bar_label_dict = dict(fmt='{:0.4f}', rotation='vertical', label_type='center', padding=3)
    else:
        bar_label_dict = eval(bar_label_dict)
    for mid, model in enumerate(models):
        mean = []
        offset = (mid - num_models / 2 + 0.5) * width
        for date in sorted(os.listdir(os.path.join(args['path'], model))):
            scenes = sorted(os.listdir(os.path.join(args['path'], model, date)))
            for idx, scene in enumerate(scenes):
                r = idx // cols
                c = int(idx - r * cols)
                data = np.loadtxt(os.path.join(args['path'], model, date, scene, args['stat_name']),
                                  skiprows=1, usecols=[2])
                if data.shape[0] == 10:
                    xticks = ['50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450',
                              '450-500', 'mean']
                elif data.shape[0] == 11:
                    xticks = ['50-100', '100-150', '150-200', '200-250', '250-300', '300-350', '350-400', '400-450',
                              '450-500', 'mavg', 'gavg']
                x = np.arange(len(xticks))
                bar = axes[r, c].bar(x - offset, data, width=width, label=model)
                axes[r, c].bar_label(bar, **bar_label_dict)
                bars.append(bar)
                label = scene.split('-')[0]
                axes[r, c].set_title(label)
                axes[r, c].set_xticks(x, xticks, rotation='vertical')
                mean.append(data[-1])
        mean_stat.update({model: np.mean(mean)})
    lines, labels = axes[0, 0].get_legend_handles_labels()
    labels = ['{}:{:.6f}'.format(label, mean_stat[label]) for label in labels]
    fig.legend(lines, labels)

    if args['quiet']:
        for k, v in mean_stat.items():
            print(k, 'mean average relative error:', v)
        plt.savefig(os.path.join(args['workspace'], args['output_name']))
    else:
        plt.show()


def center_crop(a, width=800, height=600):
    origin_height, origin_width = a.shape
    start_point = [(origin_height - height) // 2, (origin_width - width) // 2]
    end_point = [(origin_height + height) // 2, (origin_width + width) // 2]
    return misc.fit_shape(a, height, width,
                          ftype='crop_or_pad',
                          start_point=start_point,
                          end_point=end_point, mode='constant',
                          constant_values=0)


def resize(a, width=640, height=480):
    w = a.shape[-1]
    return cv2.resize(a, (width, height), interpolation=cv2.INTER_LINEAR) * (width * 1.0 / w)


def metric(args):
    assert os.path.exists(args['predicted_disparity']), args['predicted_disparity']
    assert os.path.exists(args['true_disparity']), args['true_disparity']

    import glob
    from src.dataloaders import augments as augs

    augments = augs.get(args.get('augments', []))

    if os.path.isdir(args['predicted_disparity']):
        preds = sorted(list(glob.glob(args['predicted_disparity'] + '/*')))
    else:
        with open(args['predicted_disparity'], 'r') as fp:
            preds = fp.readlines()
    if os.path.isdir(args['true_disparity']):
        trues = sorted(list(glob.glob(args['true_disparity'] + '/*')))
    else:
        with open(args['true_disparity'], 'r') as fp:
            trues = fp.readlines()
    masks = None
    if args['mask'] is not None:
        if os.path.isdir(args['mask']):
            masks = sorted(os.listdir(args['mask']))
        elif os.path.isfile(args['mask']):
            with open(args['mask'], 'r') as fp:
                masks = fp.readlines()
        else:
            raise ValueError('{} is neither file nor directory'.format(args['mask']))

    assert len(preds) == len(trues), '{} vs {}'.format(len(preds), len(trues))
    if masks is not None:
        assert len(preds) == len(masks)
    ms = metrics.get(args['metrics'])

    num_samples = len(preds)

    scale = args.get('scale', 1.0)
    # print('true disparity scale:', scale)
    shape = None

    with open(os.path.join(args['workspace'], args['output_name']), 'w') as fp:
        stat = {}
        fp.write('# filename {}\n'.format(' '.join([m for m in ms.keys()])))
        for idx, (pred, true) in enumerate(zip(preds, trues)):
            pred = pred.strip('\n')
            true = true.strip('\n')
            print('measuring [{}/{}] {}\r'.format(idx + 1, num_samples, colors.blue(os.path.basename(pred))), end='')
            p = ios.load_disp(pred, 1.0)
            t = ios.load_disp(true, scale)
            if p is None or t is None:
                logger.warning('bad file: {}, {}. ignore them'.format(pred, true))
                continue
            p = p.squeeze()
            t = augments(t.squeeze())
            if p.shape[0] != t.shape[0] or p.shape[1] != t.shape[1]:
                shape = t.shape
                p = cv2.resize(p, (t.shape[1], t.shape[0])) * (t.shape[1] * 1.0 / p.shape[1])
            f = t > 0
            if args['min_disparity'] is not None:
                f &= (t >= args['min_disparity'])
            if args['max_disparity'] is not None:
                f &= (t <= args['max_disparity'])
            res = {}
            if masks is not None:
                mask_name = masks[idx].strip('\n')
                mask = (ios.load_others(mask_name, mode='L') > 0) & f
                for n, m in ms.items():
                    res.update({n: m({'disp': p}, {'disp': t}, mask=mask)})
            else:
                for n, m in ms.items():
                    res.update({n: m({'disp': p}, {'disp': t}, mask=f)})
            for k, v in res.items():
                if k not in stat.keys():
                    stat.update({k: []})
                stat[k].append(v)

            fp.write('{},{}\n'.format(pred, ','.join('{}'.format(v) for v in res.values())))

        import termtables as tt
        header = []
        data = []
        print('\n=========================== \nmetric @{}x{}\n==========================='.format(shape[0], shape[1]))
        for k, v in stat.items():
            header.append(k)
            data.append(np.nanmean(v))
        tt.print([data], header, style=tt.styles.markdown, padding=(0, 1), alignment='c'*len(ms))


def transform(args):
    from src.misc import transformation as trans
    from src.misc import pcio
    path = os.path.join(args['workspace'], 'visualize', 'pointcloud')
    assert args['baseline'] is not None and args['intrinsics'] is not None
    dataset = ios.load_dataset(args['predicted_disparity'], args['true_disparity'], args['images'])
    logger.info(f'total {len(dataset)} samples found')
    for data in dataset:
        pred_name = data['pred']
        true_name = data['true']
        image_name = data['image']
        pred = ios.load_disp(pred_name, 1.0)
        true = ios.load_disp(true_name, args['scale'])
        image = None
        if image_name is not None:
            image = ios.load_others(image_name)
        pred_pc = trans.disparity_to_pointcloud(pred, args['baseline'], intrinsics=args['intrinsics'])
        basename = os.path.splitext(os.path.basename(pred_name))[0]
        pcio.save_pointcloud(os.path.join(path, basename + '.ply'),
                             pred_pc if image is None else np.concatenate([pred_pc, image], axis=2))
        if true is not None:
            true_pc = trans.disparity_to_pointcloud(true, args['baseline'], intrinsics=args['intrinsics'])
            pcio.save_pointcloud(os.path.join(path, basename + '_true.ply'),
                                 true_pc if image is None else np.concatenate([true_pc, image], axis=2))


def histogram(args):
    from src.dataloaders import augments as augs
    augments = augs.get(args.get('augments', []))
    for idx, disparity in enumerate(args['disparities']):
        splits = disparity.split(':')
        hists = 0
        for ix, split in enumerate(splits):
            print(split)
            h = 0
            for root, _, files in os.walk(split):
                print(split)
                for i, f in enumerate(files):
                    print('[{}/{}] {}\r'.format(i+1, len(files), f), end='')
                    disp = ios.load_disp(os.path.join(root, f), args.get('scale', 1.0))
                    disp = augments({'disp':disp})['disp']
                    hist = np.histogram(disp, bins=args['bins'], range=[args['min_disparity'], args['max_disparity']])[0]
                    h += hist / np.prod(disp.shape)
            np.savetxt(os.path.join(args['workspace'], 'histogram_{}_{}.txt'.format(idx, ix)), h)
            hists += h
        np.savetxt(os.path.join(args['workspace'], 'histogram_{}.txt'.format(idx)), hists)
        x = np.linspace(args['min_disparity'], args['max_disparity'], num=args['bins'])
        plt.plot(x, hists / hists.sum())
    plt.grid()
    plt.xlabel('disparity')
    plt.ylabel('percentage')
    plt.show()
    return hists


def view_disparity(args):
    from src.misc.open3d_window import DisparityViewer
    app = DisparityViewer(args['baseline'], args['intrinsics'],
                          path=args['predicted_disparity'], image_path=args['image_path'])
    app.run()


def analyze_pointcloud(pc: np.ndarray, args) -> list[float]:
    # filter out points by given ranges
    height_mask = (pc[..., 1] > args['height_range'][0]) & (pc[..., 1] < args['height_range'][1])
    depth_mask = (pc[..., 2] > args['depth_range'][0]) & (pc[..., 2] < args['depth_range'][1])
    mask = height_mask & depth_mask
    return misc.fit_plane(pc, mask)


def build_obstacle_map(pc: np.ndarray,
                       valid_mask: np.ndarray,
                       width_range: list[float],
                       height_range: list[float],
                       depth_range: list[float],
                       args: dict[str, typing.Any]) -> list[np.ndarray]:
    obstacle_map = None
    logger.debug('obstacle points: {}'.format(pc[valid_mask].shape[0]))
    if len(pc[valid_mask]) > 0:
        # move origin from center to top-left
        # then divide by 10 x 10 (downscale), where each pixel (grid) in grid map has the size of 10mm x 10mm
        pc[..., 0] = pc[..., 0] / args['grid_resolution']
        pc[..., 1] = -pc[..., 1] / args['grid_resolution']
        pc[..., 2] = pc[..., 2] / args['grid_resolution']
        scaled_width_range = np.asarray(width_range) / args['grid_resolution']
        scaled_height_range = np.asarray(height_range) / args['grid_resolution']
        scaled_depth_range = np.asarray(depth_range) / args['grid_resolution']

        discrete_obstacle_pc = pc.astype(np.int32)
        xmask = ((discrete_obstacle_pc[..., 0] >= scaled_width_range[0]) &
                 (discrete_obstacle_pc[..., 0] <= scaled_width_range[1]))
        ymask = ((discrete_obstacle_pc[..., 1] >= 0) &
                 (discrete_obstacle_pc[..., 1] <= scaled_height_range[1]))
        zmask = ((discrete_obstacle_pc[..., 2] >= scaled_depth_range[0]) &
                 (discrete_obstacle_pc[..., 2] <= scaled_depth_range[1]))
        valid_mask = (xmask & ymask & zmask & valid_mask)

        masked_discrete_obstacle_pc = discrete_obstacle_pc[valid_mask]
        xbins = np.arange(scaled_width_range[0], scaled_width_range[1] + 1, step=1)
        zbins = np.arange(scaled_depth_range[0], scaled_depth_range[1] + 1, step=1)
        h, w = pc.shape[0], pc.shape[1]
        plt.subplots_adjust(left=0.0, bottom=0.0, top=1.0, right=1.0)
        dpi = 100.0
        fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        fig.clear()
        if args['dimension'] == 2:
            hist, _, _ = np.histogram2d(masked_discrete_obstacle_pc[..., 0],
                                        masked_discrete_obstacle_pc[..., 2], bins=(xbins, zbins))
            obstacle_hist = hist.T
            ax = fig.gca()
            grid = ax.imshow(obstacle_hist, cmap=args['cmap'], origin='lower')
            fig.colorbar(grid, ax=ax, orientation='horizontal', fraction=0.1)
            ax.grid(which='major', axis='both', linestyle='--', linewidth=0.5)
            xticks = np.linspace(0, obstacle_hist.shape[1], xbins.shape[0])
            yticks = np.linspace(0, obstacle_hist.shape[0], zbins.shape[0])
            ax.set_xticks(xticks[::5], labels=(xbins[::5] * args['grid_resolution']).astype(np.int32),
                          rotation='vertical')
            ax.set_yticks(yticks[::5], labels=(zbins[::5] * args['grid_resolution']).astype(np.int32))
        elif args['dimension'] == 3:
            ybins = np.arange(0, scaled_height_range[1] + 1, step=1)
            hist, _ = np.histogramdd(masked_discrete_obstacle_pc.reshape(-1, 3),
                                     bins=(xbins, ybins, zbins))
            ax = fig.add_subplot(projection='3d')
            ax.view_init(elev=-args['elev'], azim=args['azim'], roll=args['roll'])
            ax.voxels(hist)
        plt.axis('equal')
        canvas = fig.canvas
        canvas.draw()
        image_flat = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        width, height = fig.get_size_inches() * fig.get_dpi()
        obstacle_map = image_flat.reshape(int(height), int(width), 3)
        plt.close(fig)
    else:
        logger.warning(colors.yellow('no obstacle found'))
    return [obstacle_map, valid_mask]


def local_obstacle_map(args):
    from src.misc import transformation as trans
    from src.misc.open3d_window import Open3DWindow
    path = os.path.join(args['workspace'], 'maps')
    os.makedirs(path, exist_ok=True)
    assert args['baseline'] is not None and args['intrinsics'] is not None
    dataset = ios.load_dataset(args['predicted_disparity'],
                               args['true_disparity'], args['images'])
    logger.info(f'total {len(dataset)} samples found')

    if args['gui']:
        win = Open3DWindow(args['baseline'], args['intrinsics'], dataset, args['width_range'], args['height_range'],
                           args['depth_range'], stereo=args['stereo'])
        win.run()

    else:
        pbar = ProgressBar(len(dataset))
        # [[mean[0,1,2], max[0,1,2], min[0,1,2], std[0,1,2]], ...]
        # shape: N x 4 x 3
        errors = []

        import termtables as tt

        for p, data in zip(pbar, dataset):  # data is a dataset sample
            pred_name = data['pred']
            true_name = data['true']
            image_name = data['image']
            pred = ios.load_disp(pred_name, 1.0).squeeze()
            image = None
            if image_name is not None:
                image = ios.load_others(image_name)
                if args['stereo']:
                    image = np.split(image, axis=1, indices_or_sections=2)[0]
                if image.shape[0] != pred.shape[0] or image.shape[1] != pred.shape[1]:
                    pred = cv2.resize(pred, (image.shape[1], image.shape[0])) * (image.shape[1] * 1.0 / pred.shape[1])
            true = None
            if true_name is not None:
                true = ios.load_disp(true_name, args['scale'])
                if true.shape[0] != pred.shape[0] or true.shape[1] != pred.shape[1]:
                    true = cv2.resize(true, (pred.shape[1], pred.shape[0])) * (pred.shape[1] * 1.0 / true.shape[1])
            # grid map
            pred_pc = trans.disparity_to_pointcloud(pred, args['baseline'], intrinsics=args['intrinsics'])

            logger.debug('before: {}, {}, {}'.format(args['width_range'], args['height_range'], args['depth_range']))

            plane = analyze_pointcloud(pred_pc, args)
            if plane is None:
                logger.warning('fitting plane failed for {}'.format(colors.yellow(pred_name)))
                continue
            rotation_matrix, mean, std, angle = plane
            logger.debug('after: {}, {}, {}'.format(args['width_range'], args['height_range'], args['depth_range']))

            base = os.path.basename(pred_name)
            pbar.set_message('mean: {}, std: {}, angle: {} <= {}'.format(colors.blue('{:0.6f}'.format(mean)),
                                                                         colors.blue('{:0.6f}'.format(std)),
                                                                         colors.blue('{:0.6f}'.format(
                                                                             angle)) if angle < 10 else colors.red(
                                                                             '{:0.6f}'.format(angle)),
                                                                         os.path.basename(pred_name)))
            shape = pred_pc.shape
            rotated_pc = (rotation_matrix @ pred_pc.reshape(-1, 3).T).T.reshape(shape)

            height_down = mean - args['mean_offset']
            height_up = - args['detect_height']
            height_range = [height_up, height_down]
            width_mask = (args['width_range'][1] > rotated_pc[..., 0]) & (rotated_pc[..., 0] > args['width_range'][0])
            # a little big higher than the floor
            # y-axis is negative, i.e., pointing to the floor
            # args['height_range'] = height_range
            height_mask = (rotated_pc[..., 1] > height_range[0]) & (
                    rotated_pc[..., 1] < height_range[1])
            depth_mask = (rotated_pc[..., 2] > args['depth_range'][0]) & (rotated_pc[..., 2] < args['depth_range'][1])

            valid_mask = (width_mask & height_mask & depth_mask)
            rotated_pc[..., 1] = rotated_pc[..., 1] - mean
            pred_obstacle_map, valid_mask = build_obstacle_map(copy.deepcopy(rotated_pc), valid_mask, args['width_range'],
                                                               height_range, args['depth_range'], args)
            true_obstacle_map = None
            if true is not None:
                true_pc = trans.disparity_to_pointcloud(true, args['baseline'], intrinsics=args['intrinsics'])
                true_rotated_pc = (rotation_matrix @ true_pc.reshape(-1, 3).T).T.reshape(shape)
                true_rotated_pc[..., 1] = true_rotated_pc[..., 1] - mean
                true_obstacle_map, valid_mask = build_obstacle_map(copy.deepcopy(true_rotated_pc),
                                                                   valid_mask,
                                                                   args['width_range'],
                                                                   height_range,
                                                                   args['depth_range'],
                                                                   args)
                true_masked_obstacles = true_rotated_pc[valid_mask]
                error = np.abs(rotated_pc[valid_mask] - true_masked_obstacles)
                errors.append([np.nanmean(error, axis=0),
                               np.nanmax(error, axis=0),
                               np.nanmin(error, axis=0),
                               np.nanstd(error, axis=0)])

            if pred_obstacle_map is not None:
                filename = os.path.join(path, '{}.png'.format(base[:-4]))
                if image is not None:
                    if args['image_mode'] == 'concat':
                        pred_image = np.split(image, 2, axis=1)[0]
                    else:
                        pred_image = copy.deepcopy(image)
                    if pred_image.shape[0] != pred.shape[0] or pred_image.shape[1] != pred.shape[1]:
                        pred_image = cv2.resize(pred_image, (pred.shape[1], pred.shape[0]),
                                                interpolation=cv2.INTER_LINEAR)
                    pred_image[valid_mask] = pred_image[valid_mask] * 0.5 + 0.5 * np.asarray([255, 0, 0])
                    pred_obstacle_map = np.concatenate([pred_image, pred_obstacle_map], axis=1)
                    if true_obstacle_map is not None:
                        pred_obstacle_map = np.concatenate([pred_obstacle_map, true_obstacle_map], axis=1)
                ios.save_others(filename, pred_obstacle_map)

        if len(errors) > 0:
            errors = np.asarray(errors)
            header = ['metric', 'X', 'Y', 'Z']
            data = [
                ['mean', np.nanmean(errors[:, 0, 0]), np.nanmean(errors[:, 0, 1]), np.nanmean(errors[:, 0, 2])],
                ['max', np.nanmean(errors[:, 1, 0]), np.nanmean(errors[:, 1, 1]), np.nanmean(errors[:, 1, 2])],
                ['min', np.nanmean(errors[:, 2, 0]), np.nanmean(errors[:, 2, 1]), np.nanmean(errors[:, 2, 2])],
                ['std', np.nanmean(errors[:, 3, 0]), np.nanmean(errors[:, 3, 1]), np.nanmean(errors[:, 3, 2])]]
            tt.print([data], header, style=tt.styles.markdown, padding=(0, 1), alignment='c'*len(header))
            with open(os.path.join(args['workspace'], 'stat'), 'w') as fp:
                fp.write('# metric X Y Z\n')
                for idx, m in zip(range(4), ['mean', 'max', 'min', 'std']):
                    fp.write('{} {} {} {}\n'.format(m, data[idx][1], data[idx][2], data[idx][3]))


def check_files(param: typing.Tuple[dict, str]) -> None:
    sample, root = param
    try:
        ios.load_others(os.path.join(root, sample['left']))
    except Exception as e:
        logger.warning(e)
        logger.warning('loading image {} failed'.format(colors.red(sample['left'])))
    try:
        ios.load_others(os.path.join(root, sample['right']))
    except Exception as e:
        logger.warning(e)
        logger.warning('loading image {} failed'.format(colors.red(sample['right'])))
    try:
        ios.load_others(os.path.join(root, sample['mask']),
                        mode='L') if 'mask' in sample.keys() else None
    except Exception as e:
        logger.warning(e)
        logger.warning('loading mask {} failed'.format(colors.red(sample['mask'])))
    try:
        ios.load_disp(os.path.join(root, sample['disp']), 1.0) if 'disp' in sample.keys() else None
    except Exception as e:
        logger.warning(e)
        logger.warning('loading disp {} failed'.format(colors.red(sample['disp'])))


def check_dataset(kwargs):
    import yaml
    import multiprocessing as mp
    config = cfg.load(kwargs['config_file'], Loader=yaml.SafeLoader)
    datasets = config['dataloader']['datasets']

    workers = kwargs.get('num_workers', mp.cpu_count())
    if workers is None:
        workers = mp.cpu_count()
    pools = mp.Pool(processes=workers)
    for k, v in datasets.items():
        logger.info('checking dataset: {}'.format(k))
        for dataset in v:
            samples = ios.load_samples(dataset['path'])
            root = dataset.get('root', '')
            logger.info('total {} samples found'.format(colors.blue(len(samples))))
            pools.apply_async(check_files, [(sample, root) for sample in samples])
    logger.info('done checking')
