'''
This script illustrates training of an inflammation classifier for patches along SI joints
'''
import argparse
import os
import shutil
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *
from pytorch_lightning.callbacks import ModelCheckpoint

from data.datasets import SPARCCDataset
from models.sparcc_cnn import Inflammation_CNN
from util.constants import *


factor = {INFLAMMATION_MODULE: 64, DEEP_INFLAMMATION_MODULE: 12, SPARCC_MODULE: 1, JOINT: 1}


def _train_module(net, train_data, val_data, args):

    train_data.mode = INFLAMMATION_MODULE
    val_data.mode = INFLAMMATION_MODULE
    train_loader = DataLoader(train_data, batch_size=factor[INFLAMMATION_MODULE]*args.train_batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=factor[INFLAMMATION_MODULE]*args.test_batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/roc-auc', mode='max')
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=args.log_refresh_rate, num_sanity_val_steps=0, deterministic=True)
    trainer.fit(net, train_loader, val_loader)

    return trainer


def _test_module(trainer, net, test_data, args):

    test_data.mode = INFLAMMATION_MODULE
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    test_loader = DataLoader(test_data, batch_size=factor[INFLAMMATION_MODULE]*args.test_batch_size,
                             num_workers=args.num_workers, pin_memory=True)
    trainer.test(net, test_loader)

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

    # network parameters
    parser.add_argument("--train_val_test_split", help="Train/validation/test split", type=str, default="0.50,0.75")
    parser.add_argument("--backbone", help="Backbone feature extractor of the model", type=str, default='ResNet18')
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
    parser.add_argument("--log_freq", help="Frequency to log results", type=int, default=50)
    parser.add_argument("--log_refresh_rate", help="Refresh rate for logging", type=int, default=1)
    parser.add_argument("--seed", help="Seed for reproducibility", type=int, default=0)

    args = parser.parse_args()
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
        else:
            args.seed = args.seed + 1
            set_seed(args.seed)

        """
            Load the data
        """
        print_frm('Loading data')
        transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), RandomDeformation(),
                             AddNoise(sigma_max=0.05)])
        train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                              args.model_checkpoint_sacrum, range_split=range_split[0], folds=args.folds, f=i,
                              train=True, transform=transform, seed=args.seed, mode=INFLAMMATION_MODULE,
                              apply_weighting=not args.omit_weighting)
        val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                            args.model_checkpoint_sacrum, range_split=range_split[1], folds=args.folds, f=i,
                            train=False, seed=args.seed, mode=INFLAMMATION_MODULE,
                            apply_weighting=not args.omit_weighting)
        print_frm('Train data distribution: Infl: %.2f - Non-infl: %.2f' % (100*np.mean(train.q_scores),
                                                                            100*np.mean(1-train.q_scores)))
        print_frm('Val data distribution: Infl: %.2f - Non-infl: %.2f' % (100*np.mean(val.q_scores),
                                                                          100*np.mean(1-val.q_scores)))
        if args.folds is None:
            test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                                 args.model_checkpoint_sacrum, range_split=range_split[2], seed=args.seed,
                                 mode=INFLAMMATION_MODULE, apply_weighting=not args.omit_weighting)
            print_frm('Test data distribution: Infl: %.2f - Non-infl: %.2f' % (100*np.mean(test.q_scores),
                                                                               100*np.mean(1-test.q_scores)))

        """
            Build the network
        """
        print_frm('Building the network')
        weights = train.score_weights[0]
        net = Inflammation_CNN(backbone=args.backbone, lr=args.lr, use_t1_input=not args.omit_t1_input,
                               use_t2_input=not args.omit_t2_input, weights=weights)
        print_frm('Balancing weights for loss function: %s' % (weights))

        """
            Train the inflammation network
        """
        print_frm('Starting training of the inflammation network')
        trainer = _train_module(net, train, val, args)
        print_frm('Testing network')
        _test_module(trainer, net, val if args.folds is not None else test, args)
        metrics.append([float(trainer.logged_metrics['test/' + m].cpu()) for m in METRICS])

        """
            Save the final model
        """
        print_frm('Saving final model')
        shutil.copyfile(trainer.checkpoint_callback.best_model_path, os.path.join(trainer.log_dir, OPTIMAL_CKPT))


    """
        Report final performance results
    """
    metrics = np.asarray(metrics)
    metrics_avg = np.mean(metrics, axis=0)
    print_frm('Final performance report:')
    print_frm('=========================')
    for i, m in enumerate(METRICS):
        print_frm('    %s: %f' % (m, metrics_avg[i]))