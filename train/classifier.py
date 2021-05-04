'''
This script illustrates training of an inflammation classifier for patches along SI joints
'''
import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from neuralnets.util.io import print_frm
from neuralnets.util.tools import set_seed
from neuralnets.util.augmentation import *
from pytorch_lightning.callbacks import ModelCheckpoint

from data.datasets import SPARCCDataset
from models.sparcc_cnn import SPARCC_CNN
from util.constants import *


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

    # optimization parameters
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=20)
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

    """
    Fix seed (for reproducibility)
    """
    set_seed(args.seed)

    """
        Load the data
    """
    print_frm('Loading data')
    split = args.train_val_test_split
    transform = Compose([Rotate90(), Flip(prob=0.5, dim=0), Flip(prob=0.5, dim=1), RandomDeformation(),
                         AddNoise(sigma_max=0.05)])
    # transform = None
    train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                          args.model_checkpoint_sacrum, range_split=(0, split[0]), transform=transform, seed=args.seed,
                          mode=INFLAMMATION_MODULE)
    val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                        range_split=(split[0], split[1]), seed=args.seed, mode=INFLAMMATION_MODULE)
    test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                         range_split=(split[1], 1), seed=args.seed, mode=INFLAMMATION_MODULE)
    freq = np.histogram(np.concatenate((train.sparcc, val.sparcc)), bins=BINS)[0]
    tmp1 = freq == 0
    tmp2 = freq != 0
    freq[tmp1] = 1
    w = 1 / (freq)
    wc = np.sum(w[tmp2])
    w_sparcc = w / wc
    print_frm('Train data distribution: Infl: %.2f - Non-infl: %.2f' % (100*np.mean(train.q_scores),
                                                                        100*np.mean(1-train.q_scores)))
    print_frm('Val data distribution: Infl: %.2f - Non-infl: %.2f' % (100*np.mean(val.q_scores),
                                                                      100*np.mean(1-val.q_scores)))
    print_frm('Test data distribution: Infl: %.2f - Non-infl: %.2f' % (100*np.mean(test.q_scores),
                                                                       100*np.mean(1-test.q_scores)))

    """
        Build the network
    """
    print_frm('Building the network')
    net = SPARCC_CNN(backbone=args.backbone, lambda_s=args.lambda_s, lr=args.lr, w_sparcc=w_sparcc)

    """
        Train the inflammation network
    """
    print_frm('Starting training of the inflammation network')
    train.mode = INFLAMMATION_MODULE
    val.mode = INFLAMMATION_MODULE
    train_loader = DataLoader(train, batch_size=64*args.train_batch_size, num_workers=args.num_workers, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=64*args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    net.set_training_mode(INFLAMMATION_MODULE)
    monitor = 'val/i-roc-auc'
    mode = 'max'
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor=monitor, mode=mode)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=args.log_refresh_rate, num_sanity_val_steps=0)
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network for inflammation prediction
    """
    print_frm('Testing network')
    test.mode = INFLAMMATION_MODULE
    test_loader = DataLoader(test, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    trainer.test(net, test_loader)

    """
        Train the intense inflammation network
    """
    print_frm('Starting training of the intense inflammation network')
    train.mode = INTENSE_INFLAMMATION_MODULE
    val.mode = INTENSE_INFLAMMATION_MODULE
    train_loader = DataLoader(train, batch_size=16*args.train_batch_size, num_workers=args.num_workers, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=4*args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    net.set_training_mode(INTENSE_INFLAMMATION_MODULE)
    monitor = 'val/ii-roc-auc'
    mode = 'max'
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor=monitor, mode=mode)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=args.log_refresh_rate, num_sanity_val_steps=0)
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network for intense inflammation prediction
    """
    print_frm('Testing network')
    test.mode = INFLAMMATION_MODULE
    test_loader = DataLoader(test, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    trainer.test(net, test_loader)

    """
        Finetune the complete network
    """
    print_frm('Starting finetuning of the complete network')
    train.mode = JOINT
    val.mode = JOINT
    train_loader = DataLoader(train, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True,
                              shuffle=True)
    val_loader = DataLoader(val, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    net.set_training_mode(JOINT)
    monitor = 'val/sim-mae'
    mode = 'min'
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor=monitor, mode=mode)
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=args.log_refresh_rate, num_sanity_val_steps=0)
    trainer.fit(net, train_loader, val_loader)

    """
        Testing the network for sparcc scoring
    """
    print_frm('Testing network')
    test.mode = JOINT
    test_loader = DataLoader(test, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)
    trainer.test(net, test_loader)
