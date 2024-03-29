'''
This script illustrates training of an inflammation classifier for patches along SI joints
'''
import os
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import roc_auc_score

from data.datasets import SPARCCDataset
from models.sparcc_cnn import DeepInflammation_CNN
from util.constants import *
from util.tools import scores
from train.sparcc_base import get_checkpoint_location


factor = {INFLAMMATION_MODULE: 64, DEEP_INFLAMMATION_MODULE: 12, SPARCC_MODULE: 1, JOINT: 1}


def _test_module(net, test_data, args, return_vote=False):

    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/roc-auc', mode='max')

    test_data.mode = DEEP_INFLAMMATION_MODULE
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=args.log_refresh_rate, num_sanity_val_steps=0, deterministic=True)
    test_loader = DataLoader(test_data, batch_size=factor[DEEP_INFLAMMATION_MODULE]*args.test_batch_size,
                             num_workers=args.num_workers, pin_memory=True)
    trainer.test(net, test_loader)

    if return_vote:
        y_pred = np.concatenate(net.y_pred['test'], axis=0)
        y_true = np.concatenate(net.y_true['test'], axis=0)
        return trainer, (y_pred, y_true)
    else:
        return trainer


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
    parser.add_argument("--repetitions", help="Number of repetitions", type=int, default=1)
    parser.add_argument("--folds", help="Number of folds (overrides repetitions parameter if provided)", type=int,
                        default=None)
    parser.add_argument("--filter-domain", help="Select a specific domain to filter out", type=str, default=None)

    # network parameters
    parser.add_argument("--train_val_test_split", help="Train/validation/test split", type=str, default=None)
    parser.add_argument("--checkpoint", help="Path to pretrained inflammation model checkpoints top directory "
                                             "(or path to the checkpoint if train_val_test_split is set)",
                        type=str, required=True)
    parser.add_argument("--backbone", help="Backbone feature extractor of the inflammation model", type=str,
                        default='ResNet18')
    parser.add_argument("--omit_t1_input", help="Boolean flag that omits usage of T1 slices", action='store_true',
                        default=False)
    parser.add_argument("--omit_t2_input", help="Boolean flag that omits usage of T1 slices", action='store_true',
                        default=False)
    parser.add_argument("--omit_weighting", help="Boolean flag that specifies ROI masking", action='store_true',
                        default=False)

    # optimization parameters
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=400)
    parser.add_argument("--lr", help="Learning rate for the optimization", type=float, default=1e-3)

    # compute parameters
    parser.add_argument("--train_batch_size", help="Batch size during training", type=int, default=1)
    parser.add_argument("--test_batch_size", help="Batch size during testing", type=int, default=1)
    parser.add_argument("--num_workers", help="Amount of workers", type=int, default=12)
    parser.add_argument("--gpus", help="Devices available for computing", type=str, default='0')
    parser.add_argument("--accelerator", help="Acceleration engine for computations", type=str, default='dp')

    # logging parameters
    parser.add_argument("--log_dir", help="Logging directory", type=str, default='logs')
    parser.add_argument("--metric_file", help="Destination to save metrics", type=str, default=None)
    parser.add_argument("--log_freq", help="Frequency to log results", type=int, default=50)
    parser.add_argument("--log_refresh_rate", help="Refresh rate for logging", type=int, default=1)
    parser.add_argument("--seed", help="Seed for reproducibility", type=int, default=0)

    args = parser.parse_args()
    if args.train_val_test_split is not None:
        args.train_val_test_split = [float(item) for item in args.train_val_test_split.split(',')]

    metrics = []
    if args.folds is not None:
        reps = args.folds
        range_split = ((0, 1), (0, 1))
    else:
        reps = args.repetitions
        f = None
        split = args.train_val_test_split
        range_split = ((0, split[1]), (0, split[1]), (split[1], 1))
    for i in range(reps):
        rep_str = 'fold' if args.folds is not None else 'repetition'
        print_frm('')
        print_frm('Start processing %s %d/%d ...' % (rep_str, i+1, reps))
        print_frm('')

        """
        Fix seed (in case of cross validation), or increment if repetitive training
        """
        if args.folds is not None:
            set_seed(args.seed)
            f = i
        else:
            args.seed = args.seed + 1
            set_seed(args.seed)

        """
            Load the data
        """
        print_frm('Loading data')
        if range_split[1][1] > 0:
            val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                args.model_checkpoint_sacrum, range_split=range_split[1], folds=args.folds, f=i,
                                train=False, seed=args.seed, mode=DEEP_INFLAMMATION_MODULE, use_t1_input=not args.omit_t1_input,
                                use_t2_input=not args.omit_t2_input, apply_weighting=not args.omit_weighting,
                                filter_domain=args.filter_domain)
        else:
            val = None
        if args.folds is None:
            test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                 args.model_checkpoint_sacrum, range_split=range_split[2], seed=args.seed,
                                 mode=DEEP_INFLAMMATION_MODULE, use_t1_input=not args.omit_t1_input,
                                 use_t2_input=not args.omit_t2_input, apply_weighting=not args.omit_weighting,
                                 filter_domain=args.filter_domain)

        """
            Build the network
        """
        print_frm('Building the network')
        if val is not None:
            weights = val.score_weights[0]
        else:
            weights = test.score_weights[0]
        net = DeepInflammation_CNN(backbone=args.backbone, lr=args.lr, use_t1_input=not args.omit_t1_input,
                                   use_t2_input=not args.omit_t2_input, weights=weights)
        ## load networks checkpoint ##
        if reps == 1 and os.path.isdir(args.checkpoint):
            y_preds = []
            y_true = None
            n_folds = len(os.listdir(os.path.join(args.checkpoint, 'lightning_logs')))
            for f in range(n_folds):
                ckpt_i_file = get_checkpoint_location(args.checkpoint, f) if f is not None else args.checkpoint
                net.load_state_dict(torch.load(ckpt_i_file)['state_dict'])

                """
                    Testing the inflammation network
                """
                print_frm('Testing network')
                trainer, (y_pred, y_true) = _test_module(net, test, args, return_vote=True)
                y_preds.append(y_pred)

            # average out predictions
            y_preds = np.mean(np.asarray(y_preds)[:, :, 1], axis=0)

            # metrics
            acs, bas, rs, ps, fprs, npvs, fs, scores_opt = scores(y_true, y_preds)
            a, ba, r, p, fpr, npv, f = scores_opt
            if np.sum(y_true) > 0:
                auc = roc_auc_score(y_true, y_preds)
            else:
                auc = 0.5
            metrics.append([a, ba, p, r, fpr, npv, f, auc, -1])
            if args.metric_file is not None:
                np.save(args.metric_file, np.stack((acs, bas, rs, ps, fprs, npvs, fs)))
        else:
            ckpt_i_file = get_checkpoint_location(args.checkpoint, f) if f is not None else args.checkpoint
            net.load_state_dict(torch.load(ckpt_i_file)['state_dict'])
            print_frm('Balancing weights for loss function: %s' % (weights))

            """
                Testing the inflammation network
            """
            print_frm('Testing network')
            trainer = _test_module(net, val if args.folds is not None else test, args)
            metrics.append([float(trainer.logged_metrics['test/' + m].cpu()) for m in METRICS])

    """
        Report final performance results
    """
    metrics = np.asarray(metrics)
    metrics_avg = np.mean(metrics, axis=0)
    print_frm('Final performance report:')
    print_frm('=========================')
    for i, m in enumerate(METRICS):
        print_frm('    %s: %f' % (m, metrics_avg[i]))
