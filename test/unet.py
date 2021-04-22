'''
This script illustrates segmentation of the illium and sacrum on a randomly selected slice
'''

import os
import argparse
import matplotlib.pyplot as plt
import torch
import cv2

from util.constants import *
from util.tools import load

from neuralnets.networks.unet import UNet2D
from neuralnets.util.visualization import overlay
from neuralnets.util.tools import normalize


if __name__ == '__main__':

    # parse all the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to the directory that contains a preprocessed dataset", type=str,
                        required=True)
    parser.add_argument("--model-checkpoint-illium", help="Path to the illium U-Net checkpoint", type=str,
                        required=True)
    parser.add_argument("--model-checkpoint-sacrum", help="Path to the sacrum U-Net checkpoint", type=str,
                        required=True)

    # network parameters
    parser.add_argument("--coi", help="Classes of interest", type=str, default="0,1")
    parser.add_argument("--fm", help="Initial amount of feature maps of the U-Net", type=int, default=16)
    parser.add_argument("--levels", help="Amount of levels of the U-Net", type=int, default=4)
    parser.add_argument("--norm", help="Type of normalization in the U-Net", type=str, default='batch')
    parser.add_argument("--activation", help="Type of activation in the U-Net", type=str, default='relu')

    args = parser.parse_args()
    args.coi = [int(item) for item in args.coi.split(',')]

    # load the dataset
    t1_data = load(os.path.join(args.data_dir, T1_PP_FILE))

    # select a random sample
    i = np.random.randint(len(t1_data))
    j = np.random.randint(len(t1_data[i]))

    # load the model
    net = UNet2D(feature_maps=args.fm, levels=args.levels, norm=args.norm, activation=args.activation, coi=args.coi)
    net.load_state_dict(torch.load(args.model_checkpoint_illium))
    net = net.cuda().eval()

    # forward propagation
    clahe = cv2.createCLAHE(clipLimit=T1_CLIPLIMIT)
    x_ = t1_data[i][j]
    x_ = normalize(x_.astype(float), type='minmax')
    x_ = clahe.apply((x_ * (2 ** 16 - 1)).astype('uint16')) / (2 ** 16 - 1)
    x = torch.from_numpy(x_[np.newaxis, np.newaxis, ...]).cuda().float()
    y_i = net(x)[0, ...].detach().cpu().numpy()
    net.load_state_dict(torch.load(args.model_checkpoint_sacrum))
    net = net.cuda().eval()
    y_s = net(x)[0, ...].detach().cpu().numpy()
    y_i = np.argmax(y_i, axis=0)
    y_s = np.argmax(y_s, axis=0)
    y = y_i
    y[y_s == 1] = 2

    # visualize sample
    x_masked = overlay(x_, y, colors=[(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    plt.imshow(x_masked)
    plt.axis('off')
    plt.show()
