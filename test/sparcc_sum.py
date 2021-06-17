'''
This script illustrates validation of the SPARCC scoring
'''
import argparse
import matplotlib.pyplot as plt

from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *

from data.datasets import SPARCCDataset
from models.sparcc_cnn import Inflammation_CNN, DeepInflammation_CNN
from util.constants import *
from train.sparcc_base import get_n_folds, get_checkpoint_location, compute_inflammation_scores, validate_sparcc_scores


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


def _process_fold(args, test_i, test_di, f=None, w_i=None, w_di=None):

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
        Compute inflammation predictions
    """
    print_frm('Computing inflammation predictions')
    y_i_test, y_di_test = compute_inflammation_scores(net_i, net_di, test_i, test_di)

    """
        Aggregate inflammation predictions to SPARCC prediction
    """
    print_frm('Aggregating inflammation predictions to SPARCC predictions')
    s_test = _aggregate_inflammation_scores(y_i_test, y_di_test)

    """
        Evaluate SPARCC scores
    """
    print_frm('Evaluating SPARCC scores')
    maes_test, maews_test, accs_test = validate_sparcc_scores(s_test, test_i.sparcc * 72)
    maews_test = maews_test.mean(axis=1)

    print_frm('Evaluation report:')
    print_frm('========================')
    print_frm('MAE: %f' % (np.min(maes_test)))
    print_frm('MAE-W: %f' % (np.min(maews_test)))
    print_frm('ACC: %f' % (np.max(accs_test)))

    # plt.plot(np.arange(0, 1, 0.001), maes_val)
    # plt.plot(np.arange(0, 1, 0.001), maes_test)
    # plt.show()
    #
    # plt.plot(np.arange(0, 1, 0.001), emds_val)
    # plt.plot(np.arange(0, 1, 0.001), emds_test)
    # plt.show()
    #
    # plt.plot(np.arange(0, 1, 0.001), accs_val)
    # plt.plot(np.arange(0, 1, 0.001), accs_test)
    # plt.show()

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

    # logging parameters
    parser.add_argument("--log_dir", help="Logging directory", type=str, default='logs')
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
    if args.fold is None and args.train_val_test_split is None:  # cross validation
        range_split = (0, 1)
        maes, maews, accs = np.zeros((folds)), np.zeros((folds)), np.zeros((folds))
        for f in range(folds):
            print_frm('')
            print_frm('Processing fold %d/%d' % (f+1, folds))
            print_frm('')
            print_frm('Loading data')
            val_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                  args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                                  use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                                  apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
            val_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                   args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                                   use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                   apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
            m, mw, a = _process_fold(args, val_i, val_di, f=f, w_i=val_i.score_weights[0], w_di=val_di.score_weights[2])
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
        val_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                              args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                              use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                              apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
        val_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                               args.model_checkpoint_sacrum, range_split=range_split, folds=folds, f=f, train=False,
                               use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                               apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
        m, mw, a = _process_fold(args, val_i, val_di, f=f, w_i=val_i.score_weights[0], w_di=val_di.score_weights[2])
    else:  # process a specific train test split
        split = args.train_val_test_split
        print_frm('Loading data')
        test_i = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                               args.model_checkpoint_sacrum, range_split=(split[1], 1),
                               use_t1_input=not args.omit_t1_input_i, use_t2_input=not args.omit_t2_input_i,
                               apply_weighting=not args.omit_weighting_i, seed=args.seed, mode=JOINT)
        test_di = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                args.model_checkpoint_sacrum, range_split=(split[1], 1),
                                use_t1_input=not args.omit_t1_input_di, use_t2_input=not args.omit_t2_input_di,
                                apply_weighting=not args.omit_weighting_di, seed=args.seed, mode=JOINT)
        maes, emds, accs = _process_fold(args, test_i, test_di, w_i=test_i.score_weights[0],
                                         w_di=test_di.score_weights[2])
