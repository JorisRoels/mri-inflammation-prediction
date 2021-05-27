'''
This script illustrates validation of the SPARCC scoring
'''
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *
from sklearn.decomposition import PCA

from data.datasets import SPARCCDataset, SPARCCClassificationDataset
from models.sparcc_cnn import Inflammation_CNN, DeepInflammation_CNN, SPARCC_MLP_Classification
from util.constants import *
from train.sparcc_base import get_n_folds, get_checkpoint_location, compute_inflammation_feature_vectors, \
    validate_sparcc_scores, class2reg


def _train_sparcc_subclassification_modules(f_train, f_val, sparcc_train, sparcc_val, args):

    f_i_train, f_ii_train = f_train
    f_i_val, f_ii_val = f_val

    nets = []
    for i in range(len(args.categories)):
        print_frm('Starting training sub-classifier %d/%d' % (i+1, len(args.categories)))

        print_frm('Setting up sub-classification (binarized) dataset')
        train = SPARCCClassificationDataset(f_i_train, f_ii_train, sparcc_train, f_red=args.f_red,
                                            categories=args.categories, k_ordinal=i)
        val = SPARCCClassificationDataset(f_i_val, f_ii_val, sparcc_val, f_red=args.f_red, categories=args.categories,
                                          k_ordinal=i)

        """
            Build the (binary) SPARCC sub-classification model
        """
        print_frm('Building the binary MLP sub-classification network')
        n_classes = 2
        net = SPARCC_MLP_Classification(lr=args.lr, f_dim=args.f_red, f_hidden=args.f_hidden, n_classes=n_classes)

        """
            Training sub-classification model
        """
        print_frm('Training sub-classification model')
        train_loader = DataLoader(train, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True,
                                  shuffle=True)
        val_loader = DataLoader(val, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
        trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                             default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                             log_every_n_steps=args.log_freq, progress_bar_refresh_rate=args.log_refresh_rate,
                             num_sanity_val_steps=0)
        trainer.fit(net, train_loader, val_loader)

        """
            Testing sub-classification model
        """
        print_frm('Testing sub-classification model')
        trainer.test(net, val_loader)

        nets.append(net)

    return nets


def _predict_sparcc_classification_module(nets, f, args):

    # get samples
    f_is, f_iis = f

    # normalize data
    f_is -= np.mean(f_is)
    f_is /= np.std(f_is)
    f_iis -= np.mean(f_iis)
    f_iis /= np.std(f_iis)

    # get dimensions
    n_samples, n_i, f_dim = f_is.shape
    _, n_ii, _ = f_iis.shape
    n_classes = len(args.categories) + 1

    # apply dimensionality reduction
    f_is = np.reshape(f_is, (-1, f_dim))
    f_is = PCA(n_components=args.f_red).fit_transform(f_is)
    f_is = np.reshape(f_is, (n_samples, n_i, args.f_red))
    f_iis = np.reshape(f_iis, (-1, f_dim))
    f_iis = PCA(n_components=args.f_red).fit_transform(f_iis)
    f_iis = np.reshape(f_iis, (n_samples, n_ii, args.f_red))

    ysb = np.zeros((len(nets), n_samples))
    ys = np.zeros((n_samples, n_classes))
    for j, net in enumerate(nets):

        # set model to GPU and evaluation mode
        net.to('cuda:0')
        net.eval()

        # compute the sub-classification scores
        for i in range(n_samples):
            # get input
            f_i = torch.from_numpy(f_is[i]).to('cuda:0').float()
            f_ii = torch.from_numpy(f_iis[i]).to('cuda:0').float()

            # reshape to correct size
            f_i = f_i.view(1, N_SLICES, N_SIDES, N_QUARTILES, f_i.size(-1))
            f_ii = f_ii.view(1, N_SLICES, N_SIDES, f_ii.size(-1))

            # compute sub-classification result
            y = torch.softmax(net(f_i, f_ii), dim=1)[0, 1]

            # save the results
            ysb[j, i] = y.detach().cpu().numpy()

    # compute final classification result
    ys[:, 0] = 1 - ysb[0, :]
    ys[:, n_classes - 1] = ysb[n_classes - 2, :]
    for i in range(1, n_classes - 1):
        ys[:, i] = ysb[i-1, :] - ysb[i, :]

    # normalize probabilities
    ys = (ys - np.min(ys)) / (np.max(ys) - np.min(ys))
    ys = ys / np.repeat(np.sum(ys, axis=1)[..., np.newaxis], 4, axis=1)

    # derive class
    ys = np.argmax(ys, axis=1)

    return ys


def _process_fold(args, train, val, test=None, f=None, w_i=None, w_ii=None):

    """
        Build the network
    """
    print_frm('Loading the pretrained inflammation classifiers')
    net_i = Inflammation_CNN(backbone=args.backbone, use_t1_input=not args.omit_t1_input,
                             use_t2_input=not args.omit_t2_input, weights=w_i)
    net_ii = DeepInflammation_CNN(backbone=args.backbone, use_t1_input=not args.omit_t1_input,
                                  use_t2_input=not args.omit_t2_input, weights=w_ii)
    ckpt_i_file = get_checkpoint_location(args.inflammation_checkpoint, f) if f is not None \
        else args.inflammation_checkpoint
    ckpt_ii_file = get_checkpoint_location(args.deep_inflammation_checkpoint, f) if f is not None \
        else args.deep_inflammation_checkpoint
    net_i.load_state_dict(torch.load(ckpt_i_file, map_location='cuda:0')['state_dict'])
    net_ii.load_state_dict(torch.load(ckpt_ii_file, map_location='cuda:0')['state_dict'])

    """
        Compute inflammation feature vectors
    """
    print_frm('Computing inflammation feature vectors')
    f_i_train, f_ii_train = compute_inflammation_feature_vectors(net_i, net_ii, train)
    f_i_val, f_ii_val = compute_inflammation_feature_vectors(net_i, net_ii, val)
    if test is not None:
        f_i_test, f_ii_test = compute_inflammation_feature_vectors(net_i, net_ii, test)

    """
        Train SPARCC sub-classification network
    """
    print_frm('Training SPARCC sub-classification network')
    nets = _train_sparcc_subclassification_modules((f_i_train, f_ii_train), (f_i_val, f_ii_val), train.sparcc,
                                                   val.sparcc, args)

    """
        Predict SPARCC classification scores on validation set
    """
    print_frm('Predicting SPARCC classification scores on validation set')
    f = (f_i_val, f_ii_val) if test is None else (f_i_test, f_ii_test)
    s_true = val.sparcc if test is None else test.sparcc
    y_pred = _predict_sparcc_classification_module(nets, f, args)

    """
        Convert class labels to regression values
    """
    print_frm('Converting class labels to regression values')
    s_pred = class2reg(y_pred, split=args.categories)

    """
        Evaluate SPARCC scores
    """
    print_frm('Evaluating SPARCC scores')
    maes_test, maews_test, accs_test = validate_sparcc_scores(s_pred[:, np.newaxis], s_true)

    print_frm('Evaluation report:')
    print_frm('========================')
    print_frm('MAE: %f' % (np.min(maes_test)))
    print_frm('MAE-W: %f' % (np.min(maews_test)))
    print_frm('ACC: %f' % (np.max(accs_test)))

    return maes_test, maews_test, accs_test


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
    parser.add_argument("--inflammation-checkpoint", help="Path to pretrained inflammation model checkpoints "
                                                          "top directory (or path to the checkpoint if "
                                                          "train_val_test_split is set)",
                        type=str, required=True)
    parser.add_argument("--inflammation-backbone", help="Backbone feature extractor of the inflammation model",
                        type=str, default='ResNet18')
    parser.add_argument("--deep-inflammation-checkpoint", help="Path to pretrained deep inflammation model "
                                                               "checkpoints top directory (or path to the "
                                                               "checkpoint if train_val_test_split is set)",
                        type=str, required=True)
    parser.add_argument("--deep-inflammation-backbone", help="Backbone feature extractor of the deep inflammation model",
                        type=str, default='ResNet18')
    parser.add_argument("--fold", help="Fold to evaluate (if not provided, an average will be provided)", type=int,
                        default=None)

    # network parameters
    parser.add_argument("--train_val_test_split", help="Train/validation/test split", type=str, default=None)
    parser.add_argument("--backbone", help="Backbone feature extractor of the model", type=str, default='ResNet18')
    parser.add_argument("--omit_t1_input", help="Boolean flag that omits usage of T1 slices", action='store_true',
                        default=False)
    parser.add_argument("--omit_t2_input", help="Boolean flag that omits usage of T1 slices", action='store_true',
                        default=False)
    parser.add_argument("--f-red", help="Dimensionality of the reduced space", type=int, default=16)
    parser.add_argument("--f-hidden", help="Dimensionality of the hidden regression layer", type=int, default=128)
    parser.add_argument("--categories", help="Categories that define the SPARCC classes", type=str, default="2,6,11")

    # optimization parameters
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=300)
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
    if args.train_val_test_split is not None:
        args.train_val_test_split = [float(item) for item in args.train_val_test_split.split(',')]
    args.categories = [int(item) for item in args.categories.split(',')]

    """
    Fix seed (for reproducibility)
    """
    set_seed(args.seed)

    """
        If cross validation, loop over all folds, otherwise perform a single run of a specific fold or train/test split
    """
    folds = get_n_folds(args.inflammation_checkpoint)
    # transform = Compose([RotateRandom(angle=10), RandomDeformation()])
    transform = None
    if args.fold is None and args.train_val_test_split is None:  # cross validation
        range_split = (0, 1)
        maes, maews, accs = np.zeros((folds)), np.zeros((folds)), np.zeros((folds))
        for f in range(folds):
            print_frm('')
            print_frm('Processing fold %d/%d' % (f+1, folds))
            print_frm('')
            print_frm('Loading data')
            train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                  args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=True,
                                  seed=args.seed, mode=JOINT, preprocess_transform=transform)
            val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                                seed=args.seed, mode=JOINT)
            m, mw, a = _process_fold(args, train, val, f=f, w_i=val.score_weights[0], w_ii=val.score_weights[2])
            maes[f], maews[f], accs[f] = np.min(m), np.min(mw), np.max(a)

        print_frm('Final evaluation report:')
        print_frm('========================')
        print_frm('MAE: %f' % (np.mean(maes)))
        print_frm('MAE-W: %f' % (np.mean(maews)))
        print_frm('ACC: %f' % (np.mean(accs)))
    elif args.fold is not None:  # process a single fold
        f = args.fold
        range_split = (0, 1)
        print_frm('')
        print_frm('Processing fold %d' % (f+1))
        print_frm('')
        print_frm('Loading data')
        train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                              args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=True,
                              seed=args.seed, mode=JOINT, preprocess_transform=transform)
        val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                            args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                            seed=args.seed, mode=JOINT)
        maes, maews, accs = _process_fold(args, val, f, w_i=val.score_weights[0], w_ii=val.score_weights[2])
    else:  # process a specific train test split
        split = args.train_val_test_split
        print_frm('Loading data')
        train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                              args.model_checkpoint_sacrum, range_split=(0, split[1]), seed=args.seed, mode=JOINT,
                              preprocess_transform=transform)
        val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                            args.model_checkpoint_sacrum, range_split=(split[0], split[1]), seed=args.seed, mode=JOINT)
        test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                             args.model_checkpoint_sacrum, range_split=(split[1], 1), seed=args.seed, mode=JOINT)
        maes, emds, accs = _process_fold(args, test, w_i=test.score_weights[0], w_ii=test.score_weights[2])
