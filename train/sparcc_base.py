'''
This file contains several useful functionalities for the various SPARCC prediction methods
'''
import os

from neuralnets.util.augmentation import *
from sklearn.metrics import accuracy_score

from util.constants import *
from util.tools import mae


def get_n_folds(data_dir):
    return len(os.listdir(os.path.join(data_dir, 'lightning_logs')))


def get_checkpoint_location(data_dir, fold):
    return os.path.join(data_dir, 'lightning_logs', 'version_' + str(fold), OPTIMAL_CKPT)


def compute_inflammation_feature_vectors(net_i, net_ii, dataset):
    # set dataset sampling mode
    dataset.mode = JOINT

    # set models to GPU and evaluation mode
    net_i.to('cuda:0')
    net_ii.to('cuda:0')
    net_i.eval()
    net_ii.eval()

    # compute the features
    f_is = []
    f_iis = []
    for i in range(len(dataset)):
        # get input
        sample = dataset[i]
        x = torch.from_numpy(sample[0]).to('cuda:0').float()

        # get shape values
        channels, n_slices, n_sides, n_quartiles, q, _ = x.size()

        # compute (deep) inflammation predictions
        x_sq = x.view(-1, channels, q, q)
        x_s = x.view(-1, channels*n_quartiles, q, q)
        f_i = torch.flatten(net_i.model.feature_extractor(x_sq), 1)
        f_ii = torch.flatten(net_ii.model.feature_extractor(x_s), 1)

        # save the results
        f_is.append(f_i.detach().cpu().numpy())
        f_iis.append(f_ii.detach().cpu().numpy())

    # stack everything together
    f_i = np.stack(f_is)
    f_ii = np.stack(f_iis)

    return f_i, f_ii


def compute_inflammation_scores(net_i, net_ii, dataset):
    # set dataset sampling mode
    dataset.mode = JOINT

    # set models to GPU and evaluation mode
    net_i.to('cuda:0')
    net_ii.to('cuda:0')
    net_i.eval()
    net_ii.eval()

    # compute the features
    y_is = []
    y_iis = []
    for i in range(len(dataset)):
        # get input
        sample = dataset[i]
        x = torch.from_numpy(sample[0]).to('cuda:0').float()

        # get shape values
        channels, n_slices, n_sides, n_quartiles, q, _ = x.size()

        # compute (deep) inflammation predictions
        x_sq = x.view(-1, channels, q, q)
        x_s = x.view(-1, channels, n_quartiles, q, q)
        y_i = net_i(x_sq)
        y_ii = net_ii(x_s)
        y_i = torch.softmax(y_i, dim=1)[:, 1]
        y_ii = torch.softmax(y_ii, dim=1)[:, 1]

        # save the results
        y_is.append(y_i.detach().cpu().numpy())
        y_iis.append(y_ii.detach().cpu().numpy())

    # stack everything together
    y_i = np.stack(y_is)
    y_ii = np.stack(y_iis)

    return y_i, y_ii


def reg2class(y, split=(2, 6, 11)):

    # reshape
    shape_orig = y.shape
    y = y.flatten()

    # convert regression values to class labels
    split = [0, *split]
    n_labels = len(split)
    y_ = np.zeros_like(y) + n_labels - 1
    for i in range(n_labels - 1):
        mask = ((y >= split[i]) * (y < split[i+1]))
        y_[mask] = i

    # reshape to original shape
    y_ = np.reshape(y_, shape_orig)

    return y_


def class2reg(y, split=(2, 6, 11)):

    # reshape
    shape_orig = y.shape
    y = y.flatten()

    # convert class labels to regression values
    split = [*split, 72]
    n_labels = len(split)
    y_ = np.zeros_like(y, dtype=float)
    m = 0
    for i in range(n_labels):
        M = split[i]
        mask = y == i
        y_[mask] = (m + M) / 2
        m = M

    # reshape to original shape
    y_ = np.reshape(y_, shape_orig)

    return y_


def validate_sparcc_scores(s_pred, s_true):

    s_pred_c = reg2class(s_pred)
    s_true_c = reg2class(s_true * 72)

    n_samples, len_t = s_pred.shape
    maes = np.zeros((len_t))
    maews = np.zeros((len_t))
    accs = np.zeros((len_t))
    for i in range(len_t):

        # evaluate metrics
        maes[i] = mae(s_true, s_pred[:, i])
        maews[i] = mae(s_true / 72, s_pred[:, i] / 72, weighted=True) * 72
        accs[i] = accuracy_score(s_true_c, s_pred_c[:, i])

    return maes, maews, accs