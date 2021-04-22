"""
    This is a script that trains a 2D U-Net to segment illium or sacrum in T1 weighted MRI data
"""

"""
    Necessary libraries
"""
import argparse
import os
import pytorch_lightning as pl

from multiprocessing import freeze_support
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from neuralnets.networks.unet import UNet2D
from neuralnets.util.augmentation import *
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed

from data.datasets import StronglyLabeledVolumesDataset


if __name__ == '__main__':
    freeze_support()

    """
        Parse all the arguments
    """
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    
    # data parameters
    parser.add_argument("--data", help="Path to the input data", required=True, type=str)
    parser.add_argument("--labels", help="Path to the labels", required=True, type=str)
    parser.add_argument("--mode", help="Class training mode (illium or sacrum)", required=True, type=str, default='illium')
    parser.add_argument("--train_val_test_split", help="Train/validation/test split of the data", type=str, default="0.50,0.75")
    parser.add_argument("--type", help="Type of the data", type=str, default="tif3d")
    parser.add_argument("--split_orientation", help="Orientation of the data splits", type=str, default='z')
    
    # network parameters
    parser.add_argument("--input_size", help="Size of the samples that propagate through the networks", type=str, default="256,256")
    parser.add_argument("--coi", help="Classes of interest", type=str, default="0,1")
    parser.add_argument("--in_channels", help="Amount of input channels for the U-Net", type=int, default=1)
    parser.add_argument("--fm", help="Initial amount of feature maps of the U-Net", type=int, default=16)
    parser.add_argument("--levels", help="Amount of levels of the U-Net", type=int, default=4)
    parser.add_argument("--dropout", help="Dropout fraction in the U-Net", type=float, default=0.10)
    parser.add_argument("--norm", help="Type of normalization in the U-Net", type=str, default='batch')
    parser.add_argument("--activation", help="Type of activation in the U-Net", type=str, default='relu')
    
    # optimization parameters
    parser.add_argument("--loss", help="Type of loss for optimization", type=str, default='ce')
    parser.add_argument("--epochs", help="Amount of epochs for training", type=int, default=100)
    parser.add_argument("--lr", help="Learning rate for training", type=float, default=0.001)
    
    # compute parameters
    parser.add_argument("--train_batch_size", help="Batch size during training", type=int, default=4)
    parser.add_argument("--test_batch_size", help="Batch size during testing", type=int, default=4)
    parser.add_argument("--num_workers", help="Amount of workers", type=int, default=12)
    parser.add_argument("--gpus", help="Devices available for computing", type=str, default='0')
    parser.add_argument("--accelerator", help="Acceleration engine for computations", type=str, default='dp')
    
    # logging parameters
    parser.add_argument("--log_dir", help="Logging directory", type=str, default='logs')
    parser.add_argument("--log_freq", help="Frequency to log results", type=int, default=50)
    parser.add_argument("--log_refresh_rate", help="Refresh rate for logging", type=int, default=-1)
    parser.add_argument("--seed", help="Seed for reproducibility", type=int, default=0)
    
    args = parser.parse_args()
    args.train_val_test_split = [float(item) for item in args.train_val_test_split.split(',')]
    args.input_size = [int(item) for item in args.input_size.split(',')]
    args.coi = [int(item) for item in args.coi.split(',')]

    """
    Fix seed (for reproducibility)
    """
    set_seed(args.seed)

    """
        Load the data
    """
    print_frm('Loading data')
    input_shape = (1, *(args.input_size))
    split = args.train_val_test_split
    transform = Compose([Flip(prob=0.5, dim=0), ContrastAdjust(adj=0.1), RandomDeformation(), AddNoise(sigma_max=0.05),
                         CleanDeformedLabels(args.coi)])
    train = StronglyLabeledVolumesDataset(args.data, args.labels, input_shape=input_shape,
                                          type=args.type, batch_size=args.train_batch_size,
                                          transform=transform, range_split=(0, split[0]),
                                          range_dir=args.split_orientation)
    val = StronglyLabeledVolumesDataset(args.data, args.labels, input_shape=input_shape, type=args.type,
                                        batch_size=args.test_batch_size, range_split=(split[0], split[1]),
                                        range_dir=args.split_orientation)
    test = StronglyLabeledVolumesDataset(args.data, args.labels, input_shape=input_shape, type=args.type,
                                         batch_size=args.test_batch_size, range_split=(split[1], 1),
                                         range_dir=args.split_orientation)
    train_loader = DataLoader(train, batch_size=args.train_batch_size, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.test_batch_size, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                             pin_memory=True)

    """
        Build the network
    """
    print_frm('Building the network')
    net = UNet2D(feature_maps=args.fm, levels=args.levels, dropout_enc=args.dropout,
                 dropout_dec=args.dropout, norm=args.norm, activation=args.activation, coi=args.coi,
                 loss_fn=args.loss, lr=args.lr)

    """
        Train the network
    """
    print_frm('Starting training')
    print_frm('Training with loss: %s' % args.loss)
    checkpoint_callback = ModelCheckpoint(save_top_k=1, verbose=True, monitor='val/mIoU', mode='max')
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, progress_bar_refresh_rate=args.log_refresh_rate,
                         callbacks=[checkpoint_callback])
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network
    """
    print_frm('Testing network')
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    trainer.test(net, test_loader)

    """
        Save the final model
    """
    os.rename(trainer.checkpoint_callback.best_model_path, 'unet-model-%s.ckpt' % args.mode)
