import cv2
import numpy as np
import os
import logging
import torch

import onnx
import onnxsim

from sigma.engine import misc
from sigma.engine.misc import ios
from sigma.core.utils import utils
from sigma.core.fontstyles import colors

logging.basicConfig(format='%(asctime)s-%(levelname)s-%(filename)s-%(funcName)s-%(lineno)d: %(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def export(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = misc.build_model(args)
    model.eval()

    if args['infer']['checkpoint'] is None:
        logger.warning(colors.yellow('checkpoint is not giving for exporting ONNX model'))
    size = model.input_size

    with torch.no_grad():
        if args['inputs'] is None:
            tof = torch.rand(1, 254, *size).to(device)
        else:
            tof = cv2.resize(ios.load_others(args['inputs']), dsize=size)
        onnx_file = args['onnx_file']
        if onnx_file is None:
            ckpt_path = args['infer']['checkpoint']
            ckpt = None
            if ckpt_path is not None:
                ckpt = os.path.basename(args['infer']['checkpoint'])
            if ckpt is None:
                onnx_file = model.tag + '.onnx'
            else:
                onnx_file = os.path.splitext(ckpt)[0] + '.onnx'
        onnx_file = os.path.join(args['workspace'], onnx_file)
        with utils.metric_it('exporting onnx to {}'.format(onnx_file)):
            torch.onnx.export(model,
                              (tof,),
                              onnx_file,
                              input_names=args['input_names'],
                              output_names=args['output_names'],
                              export_params=True,
                              opset_version=args['opset_version'])
            onnx.checker.check_model(onnx_file)

        with utils.metric_it('simplifying onnx model with onnxsim'):
            onnx_model = onnx.load(onnx_file)
            model_simp, check = onnxsim.simplify(onnx_model)
            assert check, 'simplified onnx model could not be validated'
            onnx.save(model_simp, onnx_file.replace('.onnx', '_simplified.onnx'))


def live(args):
    from ..misc.stream import Stream
    source = args.get('source', 0)
    logger.info('opening stream from camera with ID: {}'.format(colors.green(source)))
    stm = Stream(source)

    model = misc.build_model(args)
    model.cuda()
    model.eval()

    cv2.namedWindow('live', cv2.WINDOW_NORMAL)

    with torch.no_grad():
        for data, idx in stm:
            if data is None:
                continue
            left, right = np.split(data, 2, axis=1)
            left_ = torch.from_numpy(left[np.newaxis, ...].transpose(0, 3, 1, 2)).float().cuda()
            right_ = torch.from_numpy(right[np.newaxis, ...].transpose(0, 3, 1, 2)).float().cuda()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args['mixed_precision']):
                disp = model.infer(left_, right_).squeeze().cpu().numpy()

            disp_vis, _ = misc.visualize_map(disp)

            combined_img = np.hstack((left, disp_vis))
            cv2.imshow("live", combined_img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
