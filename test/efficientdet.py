#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

from effdet import create_model, create_evaluator, create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
from timm.models.layers import set_layer_config

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--annotations', required=True, type=str, help='Path to the annotated dataset')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='test',
                    help='validation split')
parser.add_argument('--model', default='tf_efficientdet_d0_mri', type=str, metavar='MODEL',
                    help='Name of model to train (default: "tf_efficientdet_d0_mri"')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=1, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 4)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--model-checkpoint', required=True, default='', type=str, metavar='PATH',
                    help='path to checkpoint to test')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=True,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=True,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='./results.json', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')
parser.add_argument('--out_dir', default='results', type=str, help='destination directory of the resulting detections')


classes = ['joint', 'joint-left', 'joint-right']

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.75
colors_pred = [(0, 0, 0), (0, 0, 255), (0, 0, 255)]
colors_target = [(0, 0, 0), (0, 255, 0), (0, 255, 0)]
thickness = 2


def draw_bbox(img, bbox_pred, bbox_target=None):

    def _draw_box(img, box, color, text=None, font=None, fontScale=None, thickness=None):
        x, y, w, h = box[:4]
        x_ = int(x + w)
        y_ = int(y + h)
        x = int(x)
        y = int(y)
        img = cv2.rectangle(img, (x, y), (x_, y_), color, thickness)

        if text is not None:
            textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
            textX = int((x + x_) / 2 - textsize[0] / 2)
            textY = int(y - textsize[1] / 2)
            img = cv2.putText(img, text, (textX, textY), font, fontScale, color, thickness, cv2.LINE_AA)

        return img

    for i in range(bbox_pred.shape[0]):
        b = int(bbox_pred[i, -1])
        img = _draw_box(img, bbox_pred[i], colors_pred[b], thickness=thickness)

        if bbox_target is not None:
            # img = _draw_box(img, bbox_target[i], colors_target[b], text=classes[b], font=font, fontScale=fontScale,
            #                 thickness=thickness)
            img = _draw_box(img, bbox_target[i], colors_target[b], thickness=thickness)

    # plt.imshow(img)
    # plt.show()
    return img


def validate(args):
    setup_default_logging()

    args.root = args.annotations
    args.checkpoint = args.model_checkpoint

    if args.amp:
        if has_apex:
            args.apex_amp = True
        elif has_native_amp:
            args.native_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    with set_layer_config(scriptable=args.torchscript):
        bench = create_model(
            args.model,
            bench_task='predict',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            redundant_bias=args.redundant_bias,
            soft_nms=args.soft_nms,
            checkpoint_path=args.checkpoint,
            checkpoint_ema=args.use_ema,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(args.dataset, args.root, args.split)
    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1
    imgs = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            for b in range(input.shape[0]):
                imgs.append(input[b].cpu().numpy())
                # targets.append(target[b].cpu().numpy())

            with amp_autocast():
                output = bench(input, img_info=target)
            evaluator.add_predictions(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0 or i == last_idx:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg)
                )

    preds = [p[:2, :] for p in evaluator.predictions]
    anns = evaluator.coco_api.imgToAnns
    targets = [np.asarray((anns[k][0]['bbox'], anns[k][1]['bbox'])) for k in range(len(imgs))]
    mean_ap = evaluator.evaluate()
    print('mAP (val set): %.2f' % (100 * mean_ap))
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    i = np.random.randint(len(imgs))
    # for i, img in enumerate(imgs):
    img = imgs[i]
    img_m = np.mean(img, axis=0)
    for c in range(3):
        img[c] = img_m
    img_ = img.transpose(1, 2, 0)
    m = img_.min()
    M = img_.max()
    img_ = ((img_ - m) / (M - m) * 255).astype('uint8').copy()
    img_ = draw_bbox(img_, preds[i], targets[i])
    plt.imshow(img_)
    plt.axis('off')
    plt.show()
    # cv2.imwrite(os.path.join(args.out_dir, '%d.jpg' % i), img_)

    return mean_ap


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()

