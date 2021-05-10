'''
This script illustrates training of an inflammation classifier for patches along SI joints
'''
import argparse

import numpy as np
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *
from pytorch_lightning.callbacks import ModelCheckpoint

from data.datasets import SPARCCDataset, SPARCCRegressionDataset
from models.sparcc_cnn import SPARCC_CNN, SPARCC_MLP
from util.constants import *


def _compute_features(model, dataset):

    # set dataset sampling mode
    dataset.mode = JOINT

    # set model to GPU and evaluation mode
    model.to('cuda:0')
    model.eval()

    # compute the features
    f_is = []
    f_iis = []
    for i in range(len(dataset)):

        # get input
        sample = dataset[i]
        x = torch.from_numpy(sample[0][np.newaxis, ...]).to('cuda:0').float()

        # get shape values
        channels = x.size(1)
        q = x.size(-1)

        # compute inflammation feature vector
        x_sq = x.view(-1, channels, q, q)
        f_i = model.feature_extractor_i(x_sq)
        f_i = torch.flatten(f_i, 1)

        # compute intense inflammation feature vector
        x_s = x.view(-1, channels * N_QUARTILES, q, q)
        f_ii = model.feature_extractor_ii(x_s)
        f_ii = torch.flatten(f_ii, 1)

        # save the results
        f_is.append(f_i.detach().cpu().numpy())
        f_iis.append(f_ii.detach().cpu().numpy())

    # stack everything together
    f_i = np.stack(f_is)
    f_ii = np.stack(f_iis)

    return f_i, f_ii


if __name__ == '__main__':

    # parse all the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to the directory that contains a preprocessed dataset", type=str,
                        required=True)
    parser.add_argument("--si-joint-model", help="Path to the SI joint detection checkpoint", type=str, required=True)
    parser.add_argument("--model-checkpoint-illium", help="Path to the illium U-Net checkpoint", type=str,
                        required=True)
    parser.add_argument("--model-checkpoint-sacrum", help="Path to the sacrum U-Net checkpoint", type=str,
                        required=True)

    # network parameters
    parser.add_argument("--train_val_test_split", help="Train/validation/test split", type=str, default="0.50,0.75")
    parser.add_argument("--backbone", help="Backbone feature extractor of the model", type=str, default='ResNet18')
    parser.add_argument("--lambda_s", help="SPARCC similarity regularization parameter", type=float, default=1e2)
    parser.add_argument("--checkpoint", help="Path to pretrained model", type=str, default='')

    # optimization parameters
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=10000)
    parser.add_argument("--lr", help="Learning rate for the optimization", type=float, default=1e-4)

    # compute parameters
    parser.add_argument("--train_batch_size", help="Batch size during training", type=int, default=64)
    parser.add_argument("--test_batch_size", help="Batch size during testing", type=int, default=64)
    parser.add_argument("--num_workers", help="Amount of workers", type=int, default=0)
    parser.add_argument("--gpus", help="Devices available for computing", type=str, default='0')
    parser.add_argument("--accelerator", help="Acceleration engine for computations", type=str, default='dp')

    # logging parameters
    parser.add_argument("--log_dir", help="Logging directory", type=str, default='logs')
    parser.add_argument("--log_freq", help="Frequency to log results", type=int, default=50)
    parser.add_argument("--log_refresh_rate", help="Refresh rate for logging", type=int, default=0)
    parser.add_argument("--seed", help="Seed for reproducibility", type=int, default=0)

    args = parser.parse_args()
    args.train_val_test_split = [float(item) for item in args.train_val_test_split.split(',')]

    """
    Fix seed (for reproducibility)
    """
    set_seed(args.seed)

    """
        Load the data
    """
    print_frm('Loading data')
    split = args.train_val_test_split
    transform = Compose([RotateRandom(angle=10), RandomDeformation()])
    train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                          args.model_checkpoint_sacrum, range_split=(0, split[0]), seed=args.seed,
                          mode=INFLAMMATION_MODULE, preprocess_transform=transform)
    val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                        range_split=(split[0], split[1]), seed=args.seed, mode=INFLAMMATION_MODULE)
    test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                         range_split=(split[1], 1), seed=args.seed, mode=INFLAMMATION_MODULE)
    freq = np.histogram(np.concatenate((train.sparcc, val.sparcc)), bins=BINS)[0]
    tmp1 = freq == 0
    tmp2 = freq != 0
    freq[tmp1] = 1
    w_sparcc = 1 / (freq)

    """
        Build the network
    """
    print_frm('Building the CNN network')
    net = SPARCC_CNN(backbone=args.backbone, lambda_s=args.lambda_s, lr=args.lr, w_sparcc=w_sparcc)
    net.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0')['state_dict'])

    """
        Compute features for the model
    """
    print_frm('Computing features')
    f_i_train, f_ii_train = _compute_features(net.model, train)
    f_i_val, f_ii_val = _compute_features(net.model, val)
    f_i_test, f_ii_test = _compute_features(net.model, test)

    """
        Build the SPARCC regression dataset
    """
    print_frm('Setting up regression dataset')
    f_hidden = 16
    train = SPARCCRegressionDataset(f_i_train, f_ii_train, train.sparcc, f_red=f_hidden)
    val = SPARCCRegressionDataset(f_i_val, f_ii_val, val.sparcc, f_red=f_hidden)
    test = SPARCCRegressionDataset(f_i_test, f_ii_test, test.sparcc, f_red=f_hidden)

    """
        Build the SPARCC regression model
    """
    print_frm('Building the MLP network')
    net = SPARCC_MLP(f_dim=f_hidden, f_hidden=128, lr=args.lr, w_sparcc=w_sparcc)

    """
        Training regression model
    """
    print_frm('Training regression model')
    train_loader = DataLoader(train, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, progress_bar_refresh_rate=args.log_refresh_rate,
                         num_sanity_val_steps=0)
    trainer.fit(net, train_loader, val_loader)

