
import numpy as np
import pickle
import math

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score


def load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def save(scores, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(scores, f)


def num2str(n, K=4):
    n_str = str(n)
    for k in range(0, K - len(n_str)):
        n_str = '0' + n_str
    return n_str


def delinearize_index(i, sz):

    inds = np.zeros_like(sz)
    d = len(sz)
    inds[d-1] = i % sz[d-1]
    i_ = i
    n_ = 1
    for j in range(d-1, 0, -1):
        i_ -= (inds[j] * n_)
        n_ *= sz[j]
        m_ = sz[j-1]
        inds[j-1] = i_ // n_ % m_

    return inds


def fpr_score(y_true, y_pred):

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    fp = np.sum((y_pred == 1) * (y_true == 0))
    n = np.sum(y_true == 0)

    return fp / n


def mae(s_true, s_pred, weighted=False, beta=10, gamma=10, delta=0.2, reduce=True):
    l1 = np.abs(s_pred - s_true)
    if weighted:
        w = 1 + np.exp(- beta * s_true) * np.exp(-gamma * l1) * l1 ** (delta)
        w /= np.sum(w)
        w *= len(l1)
    else:
        w = np.ones_like(l1)
    if reduce:
        return np.mean(w * l1)
    else:
        return w * l1


def scores(y_true, y_pred):

    # setup thresholds
    dt = 0.01
    thresholds = np.arange(0, 1, dt)

    # compute scores
    acs = np.zeros(len(thresholds))
    bas = np.zeros(len(thresholds))
    rs = np.zeros(len(thresholds))
    ps = np.zeros(len(thresholds))
    fprs = np.zeros(len(thresholds))
    fs = np.zeros(len(thresholds))
    for i, t in enumerate(thresholds):
        y_pred_label = y_pred > t
        acs[i] = accuracy_score(y_true, y_pred_label)
        bas[i] = balanced_accuracy_score(y_true, y_pred_label)
        rs[i] = recall_score(y_true, y_pred_label)
        ps[i] = precision_score(y_true, y_pred_label)
        fprs[i] = fpr_score(y_true, y_pred_label)
        fs[i] = f1_score(y_true, y_pred_label)

    # maximize scores w.r.t. f1-score
    i = np.argmax(fs)
    scores_opt = (acs[i], bas[i], rs[i], ps[i], fprs[i], fs[i])

    return acs, bas, rs, ps, fprs, fs, scores_opt


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    return qx, qy

# s_true = 0
# s_pred = np.arange(0, 1, 0.001)
# import matplotlib.pyplot as plt
#
# legend = []
#
# w, m = mae(s_true, s_pred, weighted=False)
# plt.plot(s_pred, w)
# legend.append('No-W (s_true=%.2f): mae=%.4f' % (s_true, m))
#
# for s_true in [0, 0.1, 0.25, 0.50]:
#     w, m_w = mae(s_true, s_pred, weighted=True)
#     w_, m_nw = mae(s_true, s_pred, weighted=False)
#     legend.append('W (s_true=%.2f): mae_w=%.4f, mae=%.4f' % (s_true, m_w, m_nw))
#     plt.plot(s_pred, w)
#
# plt.legend(legend)
# plt.show()