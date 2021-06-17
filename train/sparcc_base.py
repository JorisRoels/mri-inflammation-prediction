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


def compute_inflammation_feature_vectors(net_i, net_ii, ds):
    # set dataset sampling mode
    ds_i, ds_di = ds
    ds_i.mode = JOINT
    ds_di.mode = JOINT

    # set models to GPU and evaluation mode
    net_i.to('cuda:0')
    net_ii.to('cuda:0')
    net_i.eval()
    net_ii.eval()

    # compute the features
    f_is = []
    f_iis = []
    for i in range(len(ds_i)):
        # get input
        sample_i = ds_i[i]
        sample_di = ds_di[i]
        x_i = torch.from_numpy(sample_i[0]).to('cuda:0').float()
        x_di = torch.from_numpy(sample_di[0]).to('cuda:0').float()

        # get shape values
        channels_i, n_slices, n_sides, n_quartiles, q, _ = x_i.size()
        channels_di = x_di.size(0)

        # compute (deep) inflammation predictions
        x_sq = x_i.view(-1, channels_i, q, q)
        x_s = x_di.view(-1, channels_di*n_quartiles, q, q)
        f_i = torch.flatten(net_i.model.feature_extractor(x_sq), 1)
        f_ii = torch.flatten(net_ii.model.feature_extractor(x_s), 1)

        # save the results
        f_is.append(f_i.detach().cpu().numpy())
        f_iis.append(f_ii.detach().cpu().numpy())

    # stack everything together
    f_i = np.stack(f_is)
    f_ii = np.stack(f_iis)

    return f_i, f_ii


def compute_inflammation_scores(net_i, net_ii, ds_i, ds_di):
    # set dataset sampling mode
    ds_i.mode = JOINT
    ds_di.mode = JOINT

    # set models to GPU and evaluation mode
    net_i.to('cuda:0')
    net_ii.to('cuda:0')
    net_i.eval()
    net_ii.eval()

    # compute the features
    y_is = []
    y_iis = []
    for i in range(len(ds_i)):
        # get input
        sample_i = ds_i[i]
        sample_di = ds_di[i]
        x_i = torch.from_numpy(sample_i[0]).to('cuda:0').float()
        x_di = torch.from_numpy(sample_di[0]).to('cuda:0').float()

        # get shape values
        channels_i, n_slices, n_sides, n_quartiles, q, _ = x_i.size()
        channels_di = x_di.size(0)

        # compute (deep) inflammation predictions
        x_sq = x_i.view(-1, channels_i, q, q)
        x_s = x_di.view(-1, channels_di, n_quartiles, q, q)
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


def validate_sparcc_scores(s_pred, s_true, split=(2, 6, 11)):
    """
    Validate the SPARCC scores

    :param s_pred: predictions of the sparcc score (in the [0, 72] interval)
    :param s_true: ground truth sparcc scores (in the [0, 72] interval)
    :return: MAE, MAE-W and accuracy metrics
    """
    s_pred_c = reg2class(s_pred)
    s_true_c = reg2class(s_true)

    split = [0, *split]

    n_samples, len_t = s_pred.shape
    maes = np.zeros((len_t))
    maews = np.zeros((len_t, len(split)))
    accs = np.zeros((len_t))
    for i in range(len_t):

        # evaluate metrics
        maes[i] = mae(s_true, s_pred[:, i])
        for j in range(len(split)):
            if j == len(split) - 1:
                mask = (s_true >= split[j])
            else:
                mask = ((s_true >= split[j]) * (s_true < split[j + 1]))
            maews[i, j] = mae(s_true[mask], s_pred[mask, i])
        accs[i] = accuracy_score(s_true_c, s_pred_c[:, i])

    return maes, maews, accs