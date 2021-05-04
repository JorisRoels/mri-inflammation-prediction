
import numpy as np
import pickle

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


def mae(s_true, s_pred, w=1):
    return np.mean(w * np.abs(s_true - s_pred)) / np.sum(w)


def wmae(s_true, s_pred, alpha=0):

    # compute the weights
    w = np.exp(- alpha * s_true)

    # return the weighted MAE
    return mae(s_true, s_pred, w=w)


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
