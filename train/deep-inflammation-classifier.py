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
from models.sparcc_cnn import DeepInflammation_CNN, Inflammation_CNN
from util.constants import *


factor = {INFLAMMATION_MODULE: 64, INTENSE_INFLAMMATION_MODULE: 12, SPARCC_MODULE: 1, JOINT: 1}


def _train_module(net, train_data, val_data, args):

    train_data.mode = INTENSE_INFLAMMATION_MODULE
    val_data.mode = INTENSE_INFLAMMATION_MODULE
    train_loader = DataLoader(train_data, batch_size=factor[INTENSE_INFLAMMATION_MODULE]*args.train_batch_size,
                              num_workers=args.num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=factor[INTENSE_INFLAMMATION_MODULE]*args.test_batch_size,
                            num_workers=args.num_workers, pin_memory=True)
    checkpoint_callback = ModelCheckpoint(save_top_k=5, verbose=True, monitor='val/roc-auc', mode='max')
    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.gpus, accelerator=args.accelerator,
                         default_root_dir=args.log_dir, flush_logs_every_n_steps=args.log_freq,
                         log_every_n_steps=args.log_freq, callbacks=[checkpoint_callback],
                         progress_bar_refresh_rate=args.log_refresh_rate, num_sanity_val_steps=0)
    trainer.fit(net, train_loader, val_loader)

    return trainer


def _test_module(trainer, net, test_data, args):

    test_data.mode = INTENSE_INFLAMMATION_MODULE
    net.load_state_dict(torch.load(trainer.checkpoint_callback.best_model_path)['state_dict'])
    test_loader = DataLoader(test_data, batch_size=factor[INTENSE_INFLAMMATION_MODULE]*args.test_batch_size,
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
    parser.add_argument("--inflammation-checkpoint", help="Path to the inflammation classification checkpoint",
                        type=str, default=None)
    parser.add_argument("--inflammation-backbone", help="Backbone feature extractor of the inflammation model",
                        type=str, default='ResNet18')

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
    parser.add_argument("--epochs", help="Number of training epochs", type=int, default=50)
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
    train = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium,
                          args.model_checkpoint_sacrum, range_split=(0, split[0]), transform=transform, seed=args.seed,
                          mode=INTENSE_INFLAMMATION_MODULE, apply_weighting=not args.omit_weighting)
    val = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                        range_split=(split[0], split[1]), seed=args.seed, mode=INTENSE_INFLAMMATION_MODULE,
                        apply_weighting=not args.omit_weighting)
    test = SPARCCDataset(args.data_dir, args.si_joint_model, args.model_checkpoint_illium, args.model_checkpoint_sacrum,
                         range_split=(split[1], 1), seed=args.seed, mode=INTENSE_INFLAMMATION_MODULE,
                         apply_weighting=not args.omit_weighting)
    print_frm('Train data distribution: Deep infl: %.2f - No deep infl: %.2f' % (100*np.mean(train.s_scores_d),
                                                                                 100*np.mean(1-train.s_scores_d)))
    print_frm('Val data distribution: Deep infl: %.2f - No deep infl: %.2f' % (100*np.mean(val.s_scores_d),
                                                                               100*np.mean(1-val.s_scores_d)))
    print_frm('Test data distribution: Deep infl: %.2f - No deep infl: %.2f' % (100*np.mean(test.s_scores_d),
                                                                                100*np.mean(1-test.s_scores_d)))

    """
        Build the network
    """
    print_frm('Building the network')
    if args.inflammation_checkpoint is not None:
        net_i = Inflammation_CNN(backbone=args.inflammation_backbone, use_t1_input=not args.omit_t1_input,
                                 use_t2_input=not args.omit_t2_input)
        net_i.load_state_dict(torch.load(args.inflammation_checkpoint, map_location='cuda:0')['state_dict'])
        inflammation_model = net_i.model
    else:
        inflammation_model = None
    net = DeepInflammation_CNN(backbone=args.backbone, lr=args.lr, use_t1_input=not args.omit_t1_input,
                               use_t2_input=not args.omit_t2_input, inflammation_model=inflammation_model)

    """
        Train the inflammation network
    """
    print_frm('Starting training of the inflammation network')
    trainer = _train_module(net, train, val, args)
    print_frm('Testing network')
    _test_module(trainer, net, test, args)
