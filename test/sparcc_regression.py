'''
This script illustrates validation of the SPARCC scoring
'''
import argparse

import matplotlib.pyplot as plt
import numpy as np

from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *
from scipy.stats import wasserstein_distance

from data.datasets import SPARCCDataset
from models.sparcc_cnn import SPARCC_CNN
from util.constants import *
from util.tools import mae


def _compute_inflammation_scores(model, dataset):
    # set dataset sampling mode
    dataset.mode = JOINT

    # set model to GPU and evaluation mode
    model.to('cuda:0')
    model.eval()

    # compute the features
    y_is = []
    y_iis = []
    for i in range(len(dataset)):
        # get input
        sample = dataset[i]
        x = torch.from_numpy(sample[0][np.newaxis, ...]).to('cuda:0').float()

        # get shape values
        channels = x.size(1)
        q = x.size(-1)

        # compute inflammation feature vector
        y_i, y_ii, _ = model(x)
        y_i = torch.softmax(y_i, dim=1)[0, 1]
        y_ii = torch.softmax(y_ii, dim=1)[0, 1]

        # save the results
        y_is.append(y_i.detach().cpu().numpy())
        y_iis.append(y_ii.detach().cpu().numpy())

    # stack everything together
    y_i = np.stack(y_is)
    y_ii = np.stack(y_iis)

    return y_i, y_ii


def _validate_sparcc_scores(y_i, y_ii, s_true, metric, mode='min'):

    n_samples = y_i.shape[0]
    ts = np.arange(0, 1, 0.001)
    s_pred = np.zeros((n_samples, len(ts)))
    scores = np.zeros((len(ts)))
    for i, t in enumerate(ts):
        # apply thresholding
        y_i_ = (y_i > t).astype(int)
        y_ii_ = (y_ii > t).astype(int)

        # compute sparcc scores
        for j in range(n_samples):
            s = (np.sum(y_i_[j]) + np.sum(y_ii_[j]))
            s_pred[j, i] = s

        # evaluate threshold
        scores[i] = metric(s_true*72, s_pred[:, i])

    # find optimal threshold value
    if mode == 'min':
        t_opt = ts[np.argmin(scores)]
    else:
        t_opt = ts[np.argmax(scores)]

    return scores, t_opt


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
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=3000)
    parser.add_argument("--lr", help="Learning rate for the optimization", type=float, default=1e-5)

    # compute parameters
    parser.add_argument("--train_batch_size", help="Batch size during training", type=int, default=64)
    parser.add_argument("--test_batch_size", help="Batch size during testing", type=int, default=64)
    parser.add_argument("--num_workers", help="Amount of workers", type=int, default=0)
    parser.add_argument("--gpus", help="Devices available for computing", type=str, default='0')
    parser.add_argument("--accelerator", help="Acceleration engine for computations", type=str, default='dp')

    # logging parameters
    parser.add_argument("--log_dir", help="Logging directory", type=str, default='logs')
    parser.add_argument("--log_freq", help="Frequency to log results", type=int, default=50)
    parser.add_argument("--log_refresh_rate", help="Refresh rate for logging", type=int, default=1)
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
    # transform = None
    # train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
    #                       args.model_checkpoint_sacrum, range_split=(0, split[0]), seed=args.seed,
    #                       mode=INFLAMMATION_MODULE, preprocess_transform=transform)
    val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                        range_split=(split[0], split[1]), seed=args.seed, mode=INFLAMMATION_MODULE)
    test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                         range_split=(split[1], 1), seed=args.seed, mode=INFLAMMATION_MODULE)
    # freq = np.histogram(np.concatenate((val.sparcc)), bins=BINS)[0]
    # tmp1 = freq == 0
    # tmp2 = freq != 0
    # freq[tmp1] = 1
    # w_sparcc = 1 / (freq)

    """
        Build the network
    """
    print_frm('Building the CNN network')
    net = SPARCC_CNN(backbone=args.backbone, lambda_s=args.lambda_s, lr=args.lr)
    net.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0')['state_dict'])

    """
        Compute features for the model
    """
    print_frm('Computing inflammation scores')
    y_i_val, y_ii_val = _compute_inflammation_scores(net.model, val)
    y_i_test, y_ii_test = _compute_inflammation_scores(net.model, test)

    """
        Evaluate SPARCC scores
    """
    print_frm('Evaluating SPARCC scores')
    # metric = mae
    metric = wasserstein_distance
    scores_val, t_opt_val = _validate_sparcc_scores(y_i_val, y_ii_val, val.sparcc, metric)
    scores_test, t_opt_test = _validate_sparcc_scores(y_i_test, y_ii_test, test.sparcc, metric)

    plt.plot(np.arange(0, 1, 0.001), scores_val)
