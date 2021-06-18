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
from pytorch_lightning.callbacks import ModelCheckpoint

from data.datasets import SPARCCDataset, SPARCCRegressionDataset
from models.sparcc_cnn import Inflammation_CNN, DeepInflammation_CNN, SPARCC_MLP_Regression
from util.constants import *
from train.sparcc_base import get_n_folds, get_checkpoint_location, compute_inflammation_feature_vectors, validate_sparcc_scores


def _aggregate_inflammation_scores(y_i, y_ii):

    n_samples = y_i.shape[0]
    ts = np.arange(0, 1, 0.001)
    s_pred = np.zeros((n_samples, len(ts)))
    for i, t in enumerate(ts):
        # apply thresholding
        y_i_ = (y_i > t)
        y_ii_ = (y_ii > t)

        # compute sparcc scores
        for j in range(n_samples):
            s = (np.sum(y_i_[j]) + np.sum(y_ii_[j]))
            s_pred[j, i] = s

    return s_pred


def _train_sparcc_regression_module(f_train, f_val, sparcc_train, sparcc_val, args):

    f_i_train, f_ii_train = f_train
    f_i_val, f_ii_val = f_val

    print_frm('Setting up regression dataset')
    train = SPARCCRegressionDataset(f_i_train, f_ii_train, sparcc_train, f_red=args.f_red)
    val = SPARCCRegressionDataset(f_i_val, f_ii_val, sparcc_val, f_red=args.f_red)

    """
        Build the SPARCC regression model
    """
    print_frm('Building the MLP network')
    net = SPARCC_MLP_Regression(lr=args.lr, f_dim=args.f_red, f_hidden=args.f_hidden)

    """
        Training regression model
    """
    print_frm('Training regression model')
    train_loader = DataLoader(train, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/maew', mode='min')
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, progress_bar_refresh_rate=args.log_refresh_rate,
                         num_sanity_val_steps=0, callbacks=[checkpoint_callback], deterministic=True)
    trainer.fit(net, train_loader, val_loader)
    trainer.test(net, val_loader)

    return net


def _predict_sparcc_regression_module(net, f, args):

    # set model to GPU and evaluation mode
    net.to('cuda:0')
    net.eval()

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

    # apply dimensionality reduction
    f_is = np.reshape(f_is, (-1, f_dim))
    f_iis = np.reshape(f_iis, (-1, f_dim))
    if args.f_red is not None:
        f_is = PCA(n_components=args.f_red).fit_transform(f_is)
        f_iis = PCA(n_components=args.f_red).fit_transform(f_iis)
    f_is = np.reshape(f_is, (n_samples, n_i, args.f_red))
    f_iis = np.reshape(f_iis, (n_samples, n_ii, args.f_red))

    # compute the features
    ys = np.zeros((n_samples))
    for i in range(n_samples):
        # get input
        f_i = torch.from_numpy(f_is[i]).to('cuda:0').float()
        f_ii = torch.from_numpy(f_iis[i]).to('cuda:0').float()

        # reshape to correct size
        f_i = f_i.view(1, N_SLICES, N_SIDES, N_QUARTILES, f_i.size(-1))
        f_ii = f_ii.view(1, N_SLICES, N_SIDES, f_ii.size(-1))

        # compute sparcc prediction
        y = net(f_i, f_ii)[0]

        # save the results
        ys[i] = y.detach().cpu().numpy()

    return ys


def _process_fold(args, train, val, f=None, w_i=None, w_di=None):

    """
        Build the network
    """
    print_frm('Loading the pretrained inflammation classifiers')
    net_i = Inflammation_CNN(backbone=args.backbone_i, use_t1_input=not args.omit_t1_input_i,
                             use_t2_input=not args.omit_t2_input_i, weights=w_i)
    inflammation_model = net_i.model if args.ifc else None
    net_di = DeepInflammation_CNN(backbone=args.backbone_di, use_t1_input=not args.omit_t1_input_di,
                                  use_t2_input=not args.omit_t2_input_di, weights=w_di,
                                  inflammation_model=inflammation_model)
    ckpt_i_file = get_checkpoint_location(args.checkpoint_i, f) if f is not None else args.checkpoint_i
    ckpt_di_file = get_checkpoint_location(args.checkpoint_di, f) if f is not None else args.checkpoint_di
    net_i.load_state_dict(torch.load(ckpt_i_file, map_location='cuda:0')['state_dict'])
    net_di.load_state_dict(torch.load(ckpt_di_file, map_location='cuda:0')['state_dict'])

    """
        Compute inflammation feature vectors
    """
    print_frm('Computing inflammation feature vectors')
    f_train = compute_inflammation_feature_vectors(net_i, net_di, train)
    f_val = compute_inflammation_feature_vectors(net_i, net_di, val)

    """
        Train SPARCC regression network
    """
    print_frm('Training SPARCC regression network')
    net_s = _train_sparcc_regression_module(f_train, f_val, train[0].sparcc, val[0].sparcc, args)

    """
        Predict SPARCC regression scores on validation set
    """
    print_frm('Predicting SPARCC regression scores on validation set')
    s_true = val[0].sparcc
    s_pred = _predict_sparcc_regression_module(net_s, f_val, args)

    """
        Evaluate SPARCC scores
    """
    print_frm('Evaluating SPARCC scores')
    maes_test, maews_test, accs_test = validate_sparcc_scores(s_pred[:, np.newaxis] * 72, s_true * 72)

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

    # data parameters
    parser.add_argument("--train_val_test_split", help="Train/validation/test split", type=str, default=None)
    parser.add_argument("--fold", help="Fold to evaluate (if not provided, an average will be provided)", type=int,
                        default=None)

    # inflammation network parameters
    parser.add_argument("--checkpoint-i", help="Path to pretrained inflammation model checkpoints top directory "
                                               "(or path to the checkpoint if train_val_test_split is set)",
                        type=str, required=True)
    parser.add_argument("--backbone-i", help="Backbone feature extractor of the inflammation model",
                        type=str, default='ResNet18')
    parser.add_argument("--omit_t1_input-i", help="Boolean flag that omits usage of T1 slices for the inflammation model",
                        action='store_true', default=False)
    parser.add_argument("--omit_t2_input-i", help="Boolean flag that omits usage of T1 slices for the inflammation model",
                        action='store_true', default=False)
    parser.add_argument("--omit_weighting-i", help="Boolean flag that specifies ROI masking for the inflammation model",
                        action='store_true', default=False)

    # deep inflammation network parameters
    parser.add_argument("--checkpoint-di", help="Path to pretrained deep inflammation model checkpoints top directory "
                                                "(or path to the checkpoint if train_val_test_split is set)",
                        type=str, required=True)
    parser.add_argument("--backbone-di", help="Backbone feature extractor of the deep inflammation model",
                        type=str, default='ResNet18')
    parser.add_argument("--omit_t1_input-di", help="Boolean flag that omits usage of T1 slices for the deep inflammation model",
                        action='store_true', default=False)
    parser.add_argument("--omit_t2_input-di", help="Boolean flag that omits usage of T1 slices for the deep inflammation model",
                        action='store_true', default=False)
    parser.add_argument("--omit_weighting-di", help="Boolean flag that specifies ROI masking for the deep inflammation model",
                        action='store_true', default=False)
    parser.add_argument("--ifc", help="Boolean flag that specifies inflammation feature concatenation",
                        action='store_true', default=False)

    # network parameters
    parser.add_argument("--f-red", help="Dimensionality of the reduced space", type=int, default=16)
    parser.add_argument("--f-hidden", help="Dimensionality of the hidden regression layer", type=int, default=128)

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
    parser.add_argument("--log_refresh_rate", help="Refresh rate for logging", type=int, default=100)
    parser.add_argument("--seed", help="Seed for reproducibility", type=int, default=0)

    args = parser.parse_args()
    if args.train_val_test_split is not None:
        args.train_val_test_split = [float(item) for item in args.train_val_test_split.split(',')]

    """
    Fix seed (for reproducibility)
    """
    set_seed(args.seed)

    """
        If cross validation, loop over all folds, otherwise perform a single run of a specific fold or train/test split
    """
    folds = get_n_folds(args.checkpoint_i)
    # transform = Compose([RandomDeformation()])
    transform = None
    if args.fold is None and args.train_val_test_split is None:  # cross validation
        range_split = (0, 1)
        maes, maews, accs = np.zeros((folds)), np.zeros((folds)), np.zeros((folds))
        for f in range(folds):
            print_frm('')
            print_frm('Processing fold %d/%d' % (f+1, folds))
            print_frm('')
            print_frm('Loading data')
            train_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                    args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=True,
                                    use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                                    apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT,
                                    preprocess_transform=transform)
            train_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                     args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=True,
                                     use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                     apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT,
                                     preprocess_transform=transform)
            val_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                  args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                                  use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                                  apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
            val_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                   args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                                   use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                   apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
            m, mw, a = _process_fold(args, (train_i, train_di), (val_i, val_di), f=f, w_i=val_i.score_weights[0],
                                     w_di=val_di.score_weights[2])
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
        train_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=True,
                                use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                                apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
        train_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                 args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=True,
                                 use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                 apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
        val_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                              args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                              use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                              apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
        val_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                               args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                               use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                               apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
        maes, maews, accs = _process_fold(args, (train_i, train_di), (val_i, val_di), f=f, w_i=val_i.score_weights[0],
                                          w_di=val_di.score_weights[2])
    else:  # process a specific train test split
        split = args.train_val_test_split
        print_frm('Loading data')

        train_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                args.model_checkpoint_sacrum, range_split=(0, split[1]), folds=folds, f=f, train=True,
                                use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                                apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
        train_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                 args.model_checkpoint_sacrum, range_split=(0, split[1]), folds=folds, f=f, train=True,
                                 use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                 apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
        test_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                               args.model_checkpoint_sacrum, range_split=(split[1], 1), folds=folds, f=f, train=False,
                               use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                               apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
        test_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                args.model_checkpoint_sacrum, range_split=(split[1], 1), folds=folds, f=f, train=False,
                                use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
        train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                              args.model_checkpoint_sacrum, range_split=(0, split[1]), seed=args.seed, mode=JOINT,
                              preprocess_transform=transform)
        test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                             args.model_checkpoint_sacrum, range_split=(split[1], 1), seed=args.seed, mode=JOINT)
        maes, emds, accs = _process_fold(args, (train_i, train_di), (test_i, test_di), w_i=test_i.score_weights[0],
                                         w_di=test_di.score_weights[2])
