import numpy as np
import torch
import torch.utils.data as data
import os
import cv2
from tqdm import tqdm
from neuralnets.util.tools import normalize, sample_labeled_input
from neuralnets.util.io import print_frm, read_volume, mkdir
from neuralnets.data.base import slice_subset, _len_epoch
from neuralnets.data.datasets import split_segmentation_transforms
from neuralnets.networks.unet import UNet2D
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from effdet import create_model
from timm.models.layers import set_layer_config

from util.constants import *
from util.tools import load, delinearize_index, rotate
from train.sparcc_base import reg2class


class SPARCCBaseDataset(data.Dataset):
    """
    Dataset that returns feature vectors and corresponding SPARCC scores

    :param f_i: inflammation feature vectors
    :param f_ii: intense inflammation feature vectors
    :param y: SPARCC scores
    :param optional f_red: target feature dimension (T-SNE dimensionality reduction will be applied if this is set)
    """

    def __init__(self, f_i, f_ii, y, f_red=None):

        # save features and scores
        self.f_i = f_i
        self.f_ii = f_ii
        self.y = y

        # normalize data
        self.f_i -= np.mean(self.f_i)
        self.f_i /= np.std(self.f_i)
        self.f_ii -= np.mean(self.f_ii)
        self.f_ii /= np.std(self.f_ii)

        # get dimensions
        self.n_samples, self.n_i, self.f_dim = self.f_i.shape
        _, self.n_ii, _ = self.f_ii.shape

        # reduce dimensionality if necessary
        if f_red is not None:
            self.f_red = f_red

            # inflammation features
            f_i = np.reshape(self.f_i, (-1, self.f_dim))
            f_i = PCA(n_components=f_red).fit_transform(f_i)
            f_i = np.reshape(f_i, (self.n_samples, self.n_i, self.f_red))
            self.f_i = f_i

            # intense inflammation features
            f_ii = np.reshape(self.f_ii, (-1, self.f_dim))
            f_ii = PCA(n_components=f_red).fit_transform(f_ii)
            f_ii = np.reshape(f_ii, (self.n_samples, self.n_ii, self.f_red))
            self.f_ii = f_ii

            # update feature vector dimension
            self.f_dim = f_red

    def __getitem__(self, i):
        return self.f_i[i], self.f_ii[i], self.y[i]

    def __len__(self):
        return self.n_samples


class SPARCCRegressionDataset(SPARCCBaseDataset):
    """
    Dataset that returns feature vectors and corresponding SPARCC scores

    :param f_i: inflammation feature vectors
    :param f_ii: intense inflammation feature vectors
    :param y: SPARCC scores
    :param optional f_red: target feature dimension (T-SNE dimensionality reduction will be applied if this is set)
    """

    def __init__(self, f_i, f_ii, y, f_red=None):
        super().__init__(f_i, f_ii, y, f_red=f_red)


class SPARCCClassificationDataset(SPARCCBaseDataset):
    """
    Dataset that returns feature vectors and corresponding SPARCC scores

    :param f_i: inflammation feature vectors
    :param f_ii: intense inflammation feature vectors
    :param y: SPARCC scores
    :param optional f_red: target feature dimension (T-SNE dimensionality reduction will be applied if this is set)
    :param optional categories: list of thresholds that define a discrete set of categories
    :param optional k_ordinal: index that specifies the k'th case dataset for ordinal classification
    """

    def __init__(self, f_i, f_ii, y, f_red=None, categories=None, k_ordinal=None):
        super().__init__(f_i, f_ii, y, f_red=f_red)

        self.categories = categories
        self.k_ordinal = k_ordinal

        # discretize to categories if necessary
        if categories is not None:
            self.y = reg2class(self.y * 72, split=categories)

            # prep data for ordinal classification
            if k_ordinal is not None:
                self.y = self.y > k_ordinal


class SPARCCDataset(data.Dataset):
    """
    Dataset that processes the UZ MRI data for inflammation prediction

    :param data_path: path to the directory containing the merged (preprocessed) data, score and slicenumbers files
    :param si_joint_model: model that is able to locate the SI joints
    :param illum_model: model that is able to segment the illium
    :param sacrum_model: model that is able to segment the sacrum
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional transform: augmentation transformer
    :param optional seed: seed to fix the shuffling
    :param optional mode: training mode, influences the type of generated samples:
                    - INFLAMMATION_MODULE: [CHANNELS, QUARTILE_SIZE, QUARTILE_SIZE]
                    - DEEP_INFLAMMATION_MODULE: [CHANNELS, N_QUARTILES, QUARTILE_SIZE, QUARTILE_SIZE]
                    - JOINT/SPARCC_MODULE: [CHANNELS, N_SLICES, N_SIDES, N_QUARTILES, QUARTILE_SIZE, QUARTILE_SIZE]
    :param optional preprocess_transform: augmentation transformer that is applied as a preprocessing on the original
                                          slices
    :param optional apply_weighting: apply roi weighting or not
    """

    def __init__(self, data_path, si_joint_model, illum_model, sacrum_model, range_split=(0, 1), folds=None, f=None,
                 train=True, transform=None, seed=0, mode=JOINT, preprocess_transform=None, use_t1_input=True,
                 use_t2_input=True, apply_weighting=True):
        self.data_path = data_path
        self.si_joint_model = si_joint_model
        self.illum_model = illum_model
        self.sacrum_model = sacrum_model
        self.range_split = range_split
        self.folds = folds
        self.f = f
        self.train = train
        self.transform = transform
        self.seed = seed
        self.mode = mode
        self.preprocess_transform = preprocess_transform
        self.use_t1_input = use_t1_input
        self.use_t2_input = use_t2_input
        self.apply_weighting = apply_weighting

        # load all necessary files
        print_frm('    Loading pickled data')
        self.scores = load(os.path.join(data_path, SCORES_PP_FILE))
        self.slicenumbers = load(os.path.join(data_path, SLICENUMBERS_PP_FILE))
        self.t1_data = load(os.path.join(data_path, T1_PP_FILE))
        self.t2_data = load(os.path.join(data_path, T2_PP_FILE))

        # shuffle the data
        self._shuffle_data()

        # select a subset of the data
        start, stop = range_split
        start = int(start * len(self.t1_data))
        stop = int(stop * len(self.t1_data))
        self.scores = tuple([scoreset[start:stop] for scoreset in self.scores])
        self.slicenumbers = (self.slicenumbers[0][start:stop], self.slicenumbers[1][start:stop])
        self.t1_data = self.t1_data[start:stop]
        self.t2_data = self.t2_data[start:stop]

        # select fold if necessary
        if self.folds is not None and self.f is not None:
            # k-fold validation
            kf = StratifiedKFold(n_splits=self.folds)
            inds = np.arange(len(self.t1_data))
            t = 0 if self.train else 1
            y = np.zeros_like(inds)
            for i in range(len(y)):
                if mode == DEEP_INFLAMMATION_MODULE:
                    y[i] = np.sum(self.scores[1][i]) > 0
                else:
                    y[i] = np.sum(self.scores[0][i]) > 0
            inds_split = list(kf.split(inds, y))[self.f][t]
            self.scores = tuple([[scoreset[i] for i in inds_split] for scoreset in self.scores])
            self.slicenumbers = tuple([[sn[i] for i in inds_split] for sn in self.slicenumbers])
            self.t1_data = [self.t1_data[i] for i in inds_split]
            self.t2_data = [self.t2_data[i] for i in inds_split]

        # extract slices
        print_frm('    Extracting slices')
        self.t1_slices = self._extract_slices(self.t1_data, self.slicenumbers[1])
        self.t2_slices = self._extract_slices(self.t2_data, self.slicenumbers[0],
                                              target_size=self.t1_slices[0].shape[1:])

        # pre-compute SI joint locations and illium/sacrum segmentations
        t1_slices_clahe = self._compute_clahe(self.t1_slices, clip_limit=T1_CLIPLIMIT)
        print_frm('    Computing SI joint locations')
        self.si_joints = self._compute_joints(t1_slices_clahe, self.si_joint_model)
        print_frm('    Computing illium segmentation')
        self.illium = self._compute_segmentation(t1_slices_clahe, self.illum_model)
        print_frm('    Computing sacrum segmentation')
        self.sacrum = self._compute_segmentation(t1_slices_clahe, self.sacrum_model)

        # augmentation on slice level if necessary
        if self.preprocess_transform is not None:
            print_frm('    Augmenting slices...')
            self.scores, self.slicenumbers, self.t1_slices, self.t2_slices, self.si_joints, self.illium, self.sacrum = \
                self._augment_slices(self.scores, self.slicenumbers, self.t1_slices, self.t2_slices, self.si_joints,
                                     self.illium, self.sacrum)

        # extract quartiles and corresponding weight maps
        print_frm('    Extracting quartiles and weights')
        self.quartiles, self.weights = self._extract_quartiles(self.t1_slices, self.t2_slices, self.si_joints,
                                                               self.illium, self.sacrum, Q_L, Q_D)

        # compute stats for z-normalization
        self.mu = [self.quartiles[:, i, ...].mean() for i in range(self.quartiles.shape[1])]
        self.mu.append(self.weights.mean())
        self.std = [self.quartiles[:, i, ...].std() for i in range(self.quartiles.shape[1])]
        self.std.append(self.weights.std())

        # apply weighting
        if apply_weighting:
            for c in range(self.quartiles.shape[1]):
                self.quartiles[:, c, ...] = self.quartiles[:, c, ...] * self.weights

        # filter T1/T2 quartiles if necessary
        if self.use_t2_input and not self.use_t1_input:
            self.quartiles = self.quartiles[:, 1:2, ...]
        elif self.use_t1_input and not self.use_t2_input:
            self.quartiles = self.quartiles[:, 0:1, ...]

        # extract scores
        print_frm('    Extracting scores')
        self.q_scores, self.s_scores_i, self.s_scores_d, self.sparcc = self._extract_scores(self.scores)

        # compute class weights
        self.score_weights = []
        for scores in [self.q_scores, self.s_scores_i, self.s_scores_d]:
            pos = np.mean(scores)
            neg = 1 - pos
            w = [1 / neg, 1 / pos]
            self.score_weights.append(w)

    def __getitem__(self, i):
        if self.mode == INFLAMMATION_MODULE:
            j, k, l, m = delinearize_index(i, (len(self.quartiles), N_SLICES, N_SIDES, N_QUARTILES))
            quartiles = self.quartiles[j, :, k, l, m, ...]
            i_scores = self.q_scores[j, k, l, m, ...]
        elif self.mode == DEEP_INFLAMMATION_MODULE:
            j, k, l = delinearize_index(i, (len(self.quartiles), N_SLICES, N_SIDES))
            quartiles = self.quartiles[j, :, k, l, ...]
            i_scores = self.q_scores[j, k, l, ...]
            di_scores = self.s_scores_d[j, k, l]
        else:
            quartiles = self.quartiles[i]
            i_scores = self.q_scores[i]
            ii_scores = self.s_scores_i[i]
            di_scores = self.s_scores_d[i]
            s_scores = self.sparcc[i]

        # normalization
        quartiles = normalize(quartiles, type='minmax')

        # augmentation
        if self.transform is not None:
            qshape = quartiles.shape
            q = qshape[-1]
            quartiles = np.reshape(quartiles, (-1, q, q))
            quartiles = self.transform(quartiles)
            quartiles = np.reshape(quartiles, qshape)

        if self.mode == INFLAMMATION_MODULE:
            return quartiles, i_scores
        elif self.mode == DEEP_INFLAMMATION_MODULE:
            return quartiles, i_scores, di_scores
        else:
            return quartiles, i_scores, di_scores, ii_scores, s_scores

    def __len__(self):
        if self.mode == INFLAMMATION_MODULE:
            return self.quartiles.shape[0] * self.quartiles.shape[2] * self.quartiles.shape[3] * self.quartiles.shape[4]
        elif self.mode == DEEP_INFLAMMATION_MODULE:
            return self.quartiles.shape[0] * self.quartiles.shape[2] * self.quartiles.shape[3]
        else:
            return self.quartiles.shape[0]

    def _augment_slices(self, scores, slicenumbers, t1_data, t2_data, si_joints, illium, sacrum):

        n = len(t1_data)
        transform = self.preprocess_transform
        ds1 = []
        ds2 = []
        dsi = []
        dss = []
        dssi = []
        for i in tqdm(range(n), desc='Augmenting slices'):

            # get the data
            x_t1 = t1_data[i]
            x_t2 = t2_data[i]
            x_i = (illium[i] * (2**16-1)).astype('uint16')
            x_s = (sacrum[i] * (2**16-1)).astype('uint16')
            sn1, sn2 = slicenumbers[0][i], slicenumbers[1][i]
            score = tuple([s[i] for s in scores])
            si_joint = si_joints[i]

            # reshape to appropriate size
            n_slices, sz_orig, _ = x_t2.shape
            x = np.concatenate((x_t1, x_t2, x_i, x_s), axis=0).astype(float)

            # augment sample
            data_t1 = []
            data_t2 = []
            data_i = []
            data_s = []
            data_si = []
            for j in range(REPS):

                # score invariant augmentation
                x_ = transform(x)
                x_t1_, x_t2_, x_i_, x_s_ = np.split(x_, (x_t1.shape[0], x_t1.shape[0]+x_t2.shape[0], x_t1.shape[0]+x_t2.shape[0]+x_i.shape[0]))

                score_ = score
                si_joint_ = si_joint

                # apply flips
                if np.random.rand() < 0.5:
                    # flip
                    x_t1_ = x_t1_[:, :, ::-1]
                    x_t2_ = x_t2_[:, :, ::-1]
                    x_i_ = x_i_[:, :, ::-1]
                    x_s_ = x_s_[:, :, ::-1]

                    # adjust scores (flip sides)
                    for k in range(len(score_)):
                        s_tmp = score_[k][:, 0].copy()
                        score_[k][:, 0] = score_[k][:, 1]
                        score_[k][:, 1] = s_tmp

                    # adjust joint
                    s_tmp = si_joint_[:, 0, :].copy()
                    si_joint_[:, 0, :] = si_joint_[:, 1, :]
                    si_joint_[:, 1, :] = s_tmp

                # apply rotation
                if np.random.rand() < 0.5:
                    # rotate
                    center = (x_t1_.shape[1] // 2, x_t1_.shape[2] // 2)
                    angle = (2 * MAX_ANGLE * (np.random.rand() - 0.5))
                    angle_r = angle / 180 * np.pi
                    R = cv2.getRotationMatrix2D(center, angle, 1)
                    for k in range(n_slices):
                        x_t1_[k] = cv2.warpAffine(x_t1_[k], R, x_t1_.shape[1:])
                        x_t2_[k] = cv2.warpAffine(x_t2_[k], R, x_t2_.shape[1:])
                        x_i_[k] = cv2.warpAffine(x_i_[k], R, x_i_.shape[1:])
                        x_s_[k] = cv2.warpAffine(x_s_[k], R, x_s_.shape[1:])

                    # scores don't need adjustment

                    # adjust joint
                    for k in range(si_joint_.shape[0]):
                        si_joint_[k, 0, :2] = rotate(center, si_joint_[k, 0, :2], -angle_r)
                        si_joint_[k, 0, 2:] = rotate(center, si_joint_[k, 0, 2:], -angle_r)
                        si_joint_[k, 1, :2] = rotate(center, si_joint_[k, 1, :2], -angle_r)
                        si_joint_[k, 1, 2:] = rotate(center, si_joint_[k, 1, 2:], -angle_r)

                # extend data
                data_t1.append(np.maximum(0, x_t1_).astype('uint16'))
                data_t2.append(np.maximum(0, x_t2_).astype('uint16'))
                data_i.append(np.minimum(1, np.maximum(0, x_i_ / (2**16-1))))
                data_s.append(np.minimum(1, np.maximum(0, x_s_ / (2**16-1))))
                for k in range(len(scores)):
                    scores[k].append(score_[k])
                slicenumbers[0].append(sn1)
                slicenumbers[1].append(sn2)
                data_si.append(si_joint_)

            # append data
            ds1.append(np.asarray(data_t1, dtype='uint16'))
            ds2.append(np.asarray(data_t2, dtype='uint16'))
            dsi.append(np.asarray(data_i))
            dss.append(np.asarray(data_s))
            dssi.append(np.asarray(data_si))

        # concatenate data
        t1_data = np.concatenate([t1_data, *ds1], axis=0)
        t2_data = np.concatenate([t2_data, *ds2], axis=0)
        illium = np.concatenate([illium, *dsi], axis=0)
        sacrum = np.concatenate([sacrum, *dss], axis=0)
        si_joints = np.concatenate([si_joints, *dssi], axis=0)

        return scores, slicenumbers, t1_data, t2_data, si_joints, illium, sacrum

    def _shuffle_data(self):
        np.random.seed(self.seed)
        inds_shuffled = np.random.permutation(np.arange(len(self.t1_data)))
        self.scores = tuple([[scoreset[i] for i in inds_shuffled] for scoreset in self.scores])
        self.slicenumbers = ([self.slicenumbers[0][i] for i in inds_shuffled],
                             [self.slicenumbers[1][i] for i in inds_shuffled])
        self.t1_data = [self.t1_data[i] for i in inds_shuffled]
        self.t2_data = [self.t2_data[i] for i in inds_shuffled]

    def _extract_slices(self, data, slicenumbers, target_size=None):

        # initialize slices
        data_size = data[0].shape[1:]
        if target_size is None:
            slices = np.zeros((len(data), N_SLICES, data_size[0], data_size[1]), dtype='uint16')
        else:
            slices = np.zeros((len(data), N_SLICES, target_size[0], target_size[1]), dtype='uint16')

        # extract slices
        for i in range(len(data)):
            for j in range(N_SLICES):
                s = slicenumbers[i][j]
                if target_size is None:
                    slices[i, j, ...] = data[i][s]
                else:
                    # upsample to the target size
                    slices[i, j, ...] = cv2.resize(data[i][s], dsize=target_size, interpolation=cv2.INTER_CUBIC)

        return slices

    def _compute_clahe(self, data, clip_limit=T1_CLIPLIMIT):

        clahe = cv2.createCLAHE(clipLimit=clip_limit)
        data = data.astype(float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x = normalize(data[i, j, ...], type='minmax')
                data[i, j, ...] = clahe.apply((x * (2 ** 16 - 1)).astype('uint16')) / (2 ** 16 - 1)

        return data

    def _filter_bboxes(self, bboxes, sz):

        def bbox_is_left(bbox, sz):
            x_max, y_max = sz
            x, y, x_, y_ = bbox[:4]
            return x > x_max // 2 and x_ < x_max and y >= 0 and y_ < y_max

        def quartiles_exceed_edges_left(bbox, q, sz):
            x_max, y_max = sz
            x, y, x_, y_ = bbox[:4]
            x_j = (x + x_) // 2
            y_j = (y + y_) // 2
            return x_j + q > x_max or y_j - q < 0 or y_j + q > y_max

        def bbox_is_right(bbox, sz):
            x_max, y_max = sz
            x, y, x_, y_ = bbox[:4]
            return x < x_max // 2 and x_ >= 0 and y >= 0 and y_ < y_max

        def quartiles_exceed_edges_right(bbox, q, sz):
            x_max, y_max = sz
            x, y, x_, y_ = bbox[:4]
            x_j = (x + x_) // 2
            y_j = (y + y_) // 2
            return x_j - q < 0 or y_j - q < 0 or y_j + q > y_max

        q = (Q_L + 2*Q_D)
        left_bboxes = [(-1, -1, -1, -1, 0, 1)]
        right_bboxes = [(-1, -1, -1, -1, 0, 1)]
        for bbox in bboxes:
            if bbox_is_left(bbox[:4], sz) and not quartiles_exceed_edges_left(bbox[:4], q, sz):
                left_bboxes.append(bbox)
            elif bbox_is_right(bbox[:4], sz) and not quartiles_exceed_edges_right(bbox[:4], q, sz):
                right_bboxes.append(bbox)

        left_bboxes = np.reshape(np.asarray(left_bboxes), (-1, 6))
        right_bboxes = np.reshape(np.asarray(right_bboxes), (-1, 6))

        return left_bboxes, right_bboxes

    def _maximize_bbox_score(self, bboxes):

        scores = bboxes[:, 4]
        i_max = np.argmax(scores)

        return bboxes[i_max]

    def _post_process_joint_bboxes(self, boxes):

        n, n_slices, n_sides, n_coos = boxes.shape

        # fix unknown detections
        boxes_new = np.zeros_like(boxes)
        bbox_med = np.zeros((n, n_sides, n_coos))
        for i in range(n):
            for j in range(n_sides):
                for k in range(n_coos):
                    c = boxes[i, :, j, k]
                    c_pos = c[c >= 0]
                    bbox_med[i, j, k] = np.median(c_pos)
                    boxes_new[i, :, j, k][c >= 0] = c_pos
                    boxes_new[i, :, j, k][c < 0] = bbox_med[i, j, k]

        # fix detections that are too far away from the median
        for i in range(n):
            for k in range(n_sides):
                j_med = (bbox_med[i, k, :2] + bbox_med[i, k, 2:4]) / 2
                for j in range(n_slices):
                    j_pred = (boxes_new[i, j, k, :2] + boxes_new[i, j, k, 2:4]) / 2
                    d = np.sqrt(np.sum((j_med - j_pred)**2))
                    if d > MEDIAN_THRESHOLD:
                        boxes_new[i, j, k, :] = bbox_med[i, k, :]

        return boxes

    def _compute_joints(self, data, model, out_file=None):

        # load the joint model
        with set_layer_config(scriptable=False):
            bench = create_model(
                'tf_efficientdet_d0_mri',
                bench_task='predict',
                num_classes=1,
                redundant_bias=None,
                soft_nms=None,
                checkpoint_path=model,
            )
        bench = bench.cuda()
        amp_autocast = torch.cuda.amp.autocast
        bench.eval()

        # forward propagation
        bboxes = np.zeros((data.shape[0], N_SLICES, 2, 4))
        with torch.no_grad():
            for i in tqdm(range(data.shape[0]), desc='Computing SI joint locations'):
                x = data[i]
                x = (x - X_MU) / X_STD
                input = torch.from_numpy(np.repeat(x[:, np.newaxis, ...], 3, axis=1)).cuda()
                with amp_autocast():
                    output = bench(input.float()).cpu().numpy()
                for j in range(N_SLICES):
                    bb_l, bb_r = self._filter_bboxes(output[j], x.shape[1:])
                    bboxes[i, j, 0] = self._maximize_bbox_score(bb_l)[:4]
                    bboxes[i, j, 1] = self._maximize_bbox_score(bb_r)[:4]

        # post-process joint bboxes
        bboxes = self._post_process_joint_bboxes(bboxes)

        # save to tmp dir if necessary
        if out_file is not None:
            mkdir(os.path.dirname(out_file))
            np.save(out_file, bboxes)

        return bboxes

    def _compute_segmentation(self, data, model, out_file=None):

        # load the segmentation model
        net = UNet2D(feature_maps=FM, levels=LEVELS, norm=NORM, activation=ACTIVATION, coi=COI)
        net.load_state_dict(torch.load(model))
        net = net.cuda().eval()

        # forward propagation
        segmentation = np.zeros_like(data)
        with torch.no_grad():
            for i in tqdm(range(data.shape[0]), desc='Computing segmentation'):
                x = data[i]
                input = torch.from_numpy(x[:, np.newaxis, ...]).cuda()
                output = torch.softmax(net(input.float()), dim=1).cpu().numpy()
                segmentation[i, ...] = output[:, 1, ...]

        # save to tmp dir if necessary
        if out_file is not None:
            mkdir(os.path.dirname(out_file))
            np.save(out_file, segmentation)

        return segmentation

    def _compute_weights(self, s):
        # s_exp = W_0 * np.exp(W_S * s)
        # z = np.sum(s_exp)
        s_exp = s
        z = 1
        return s_exp / z

    def _synced_extraction(self, x_1, x_2, bbox, p_i, p_s, q_l, q_d, s):

        # padding
        pad = int(np.sqrt(2) * (q_l + 2 * q_d))
        target_shape = (2*pad + x_1.shape[0], 2*pad + x_1.shape[1])

        # bounding box coordinates
        i_ul, j_ul, i_br, j_br = bbox

        # joint location
        i_j = int(pad + (i_ul + i_br) // 2)
        j_j = int(pad + (j_ul + j_br) // 2)

        # joint angle
        alpha_j = np.arctan((i_br - i_ul) / (j_br - j_ul))

        # rotate image to fix joint region
        sgn = 1 if s == 0 else -1
        R = cv2.getRotationMatrix2D((i_j, j_j), (sgn * alpha_j) / np.pi * 180, 1)
        x_1_r = cv2.warpAffine(np.pad(x_1, pad), R, target_shape)
        x_2_r = cv2.warpAffine(np.pad(x_2, pad), R, target_shape)
        p_i_r = cv2.warpAffine(np.pad(p_i, pad), R, target_shape)
        p_s_r = cv2.warpAffine(np.pad(p_s, pad), R, target_shape)
        x_r = np.stack((x_1_r, x_2_r, p_i_r, p_s_r))

        # extract patches
        p = q_l + 2 * q_d
        if s == 0:
            ds = ((1, -1), (-1, -1), (-1, 1), (1, 1))
        else:
            ds = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        q_ = np.zeros((x_r.shape[0], N_QUARTILES, p, p))
        for k, (di, dj) in enumerate(ds):
            i0, i1 = i_j-di*q_d, i_j+di*(q_l+q_d)
            j0, j1 = j_j-dj*q_d, j_j+dj*(q_l+q_d)
            i_start, i_stop = min(i0, i1), max(i0, i1)
            j_start, j_stop = min(j0, j1), max(j0, j1)
            q_[:, k, ...] = x_r[:, j_start:j_stop, i_start:i_stop]
        q = q_[:2, ...].astype('uint16')
        s = q_[2, ...]
        s[1:3] = q_[3, 1:3, ...]

        # compute normalized weights from segmentation
        w = self._compute_weights(s)

        return q, w

    def _extract_quartiles(self, t1_slices, t2_slices, si_joints, illium, sacrum, q_l, q_d, out_file=None):

        # allocate space for quartiles and corresponding weight maps
        n = len(t1_slices)
        p = q_l + 2 * q_d
        quartiles = np.zeros((n, 2, N_SLICES, N_SIDES, N_QUARTILES, p, p), dtype='uint16')
        weights = np.zeros((n, N_SLICES, N_SIDES, N_QUARTILES, p, p))

        # extract quartiles and weight maps
        for i in range(n):
            for j in range(N_SLICES):
                for k in range(N_SIDES):
                    q, w = self._synced_extraction(t1_slices[i, j], t2_slices[i, j], si_joints[i, j, k], illium[i, j],
                                                   sacrum[i, j], q_l, q_d, k)
                    quartiles[i, :, j, k] = q
                    weights[i, j, k] = w

        # save to tmp dir if necessary
        if out_file is not None:
            mkdir(os.path.dirname(out_file[0]))
            np.save(out_file[0], quartiles)
            np.save(out_file[1], weights)

        return quartiles, weights

    def _extract_scores(self, scores):

        n = len(scores[0])

        q_scores = np.asarray(scores[0], dtype=int)
        s_scores_d = np.asarray(scores[1], dtype=int)
        s_scores_i = np.asarray(scores[2], dtype=int)
        sparcc = np.sum(q_scores.reshape(n, -1), axis=1) + np.sum(s_scores_i.reshape(n, -1), axis=1) + \
                 np.sum(s_scores_i.reshape(n, -1), axis=1)
        sparcc = sparcc / 72

        return q_scores, s_scores_i, s_scores_d, sparcc


class InflammationQuartileDataset(data.Dataset):
    """
    Dataset that processes the UZ MRI data for inflammation prediction

    :param data_path: path to the directory containing the merged (preprocessed) data, score and slicenumbers files
    :param si_joint_model: model that is able to locate the SI joints
    :param illum_model: model that is able to segment the illium
    :param sacrum_model: model that is able to segment the sacrum
    :param optional q_l: size of the quartiles
    :param optional q_d: margin of the quartiles
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional transform: augmentation transformer
    """

    def __init__(self, data_path, si_joint_model, illum_model, sacrum_model, q_l=64, q_d=10, range_split=(0, 1),
                 transform=None):
        self.data_path = data_path
        self.si_joint_model = si_joint_model
        self.illum_model = illum_model
        self.sacrum_model = sacrum_model
        self.q_l = q_l
        self.q_d = q_d
        self.range_split = range_split
        self.transform = transform

        # load all necessary files
        self.scores = load(os.path.join(data_path, SCORES_PP_FILE))
        self.slicenumbers = load(os.path.join(data_path, SLICENUMBERS_PP_FILE))
        self.t1_data = load(os.path.join(data_path, T1_PP_FILE))
        self.t2_data = load(os.path.join(data_path, T2_PP_FILE))

        # shuffle the data
        self._shuffle_data()

        # select a subset of the data
        start, stop = range_split
        start = int(start * len(self.t1_data))
        stop = int(stop * len(self.t1_data))
        self.scores = tuple([scoreset[start:stop] for scoreset in self.scores])
        self.slicenumbers = (self.slicenumbers[0][start:stop], self.slicenumbers[1][start:stop])
        self.t1_data = self.t1_data[start:stop]
        self.t2_data = self.t2_data[start:stop]

        # extract slices
        self.t1_slices = self._extract_slices(self.t1_data, self.slicenumbers[1])
        self.t2_slices = self._extract_slices(self.t2_data, self.slicenumbers[0],
                                              target_size=self.t1_slices[0].shape[1:])

        # pre-compute SI joint locations and illium/sacrum segmentations
        SI_JOINTS_TMP_FILE = os.path.join(CACHE_DIR, '%s_%d_%d.%s' % (SI_JOINTS_TMP, start, stop, EXT))
        SEG_I_TMP_FILE = os.path.join(CACHE_DIR, '%s_%d_%d.%s' % (SEG_I_TMP, start, stop, EXT))
        SEG_S_TMP_FILE = os.path.join(CACHE_DIR, '%s_%d_%d.%s' % (SEG_S_TMP, start, stop, EXT))
        if not os.path.exists(SI_JOINTS_TMP_FILE) or \
           not os.path.exists(SEG_I_TMP_FILE) or \
           not os.path.exists(SEG_S_TMP_FILE):
            t1_slices_clahe = self._compute_clahe(self.t1_slices, clip_limit=T1_CLIPLIMIT)
        if os.path.exists(SI_JOINTS_TMP_FILE):
            self.si_joints = np.load(SI_JOINTS_TMP_FILE)
        else:
            self.si_joints = self._compute_joints(t1_slices_clahe, self.si_joint_model, out_file=SI_JOINTS_TMP_FILE)
        if os.path.exists(SEG_I_TMP_FILE):
            self.illium = np.load(SEG_I_TMP_FILE)
        else:
            self.illium = self._compute_segmentation(t1_slices_clahe, self.illum_model, out_file=SEG_I_TMP_FILE)
        if os.path.exists(SEG_S_TMP_FILE):
            self.sacrum = np.load(SEG_S_TMP_FILE)
        else:
            self.sacrum = self._compute_segmentation(t1_slices_clahe, self.sacrum_model, out_file=SEG_S_TMP_FILE)

        # extract quartiles and corresponding weight maps
        Q_TMP_FILE = os.path.join(CACHE_DIR, '%s_%d_%d.%s' % (Q_TMP, start, stop, EXT))
        W_TMP_FILE = os.path.join(CACHE_DIR, '%s_%d_%d.%s' % (W_TMP, start, stop, EXT))
        if os.path.exists(Q_TMP_FILE) and os.path.exists(W_TMP_FILE):
            self.quartiles = np.load(Q_TMP_FILE)
            self.weights = np.load(W_TMP_FILE)
        else:
            self.quartiles, self.weights = self._extract_quartiles(self.t1_slices, self.t2_slices, self.si_joints,
                                                                   self.illium, self.sacrum, self.q_l, self.q_d,
                                                                   out_file=(Q_TMP_FILE, W_TMP_FILE))

        # compute stats for z-normalization
        self.mu = [self.quartiles[:, i, ...].mean() for i in range(self.quartiles.shape[1])]
        self.mu.append(self.weights.mean())
        self.std = [self.quartiles[:, i, ...].std() for i in range(self.quartiles.shape[1])]
        self.std.append(self.weights.std())

        # extract scores
        self.q_scores, self.s_scores_i, self.s_scores_d, self.sparcc = self._extract_scores(self.scores)

    def __getitem__(self, i):
        quartiles = normalize(self.quartiles[i], type='minmax')
        weighted_quartiles = np.concatenate((quartiles, self.weights[i][np.newaxis, ...]))

        # augmentation
        if self.transform is not None:
            c, n_slices, n_sides, n_quartiles, q, _ = weighted_quartiles.shape
            for slice in range(n_slices):  # apply augmentations on a slice level
                wq = np.reshape(weighted_quartiles[:, slice, ...], (-1, q, q))
                wq = self.transform(wq)
                weighted_quartiles[:, slice, ...] = np.reshape(wq, (c, n_sides, n_quartiles, q, q))

        return weighted_quartiles, self.q_scores[i], self.s_scores_i[i], self.s_scores_d[i], self.sparcc[i]

    def __len__(self):
        return len(self.quartiles)

    def _shuffle_data(self):
        inds_shuffled = np.random.permutation(np.arange(len(self.t1_data)))
        self.scores = tuple([[scoreset[i] for i in inds_shuffled] for scoreset in self.scores])
        self.slicenumbers = ([self.slicenumbers[0][i] for i in inds_shuffled],
                             [self.slicenumbers[1][i] for i in inds_shuffled])
        self.t1_data = [self.t1_data[i] for i in inds_shuffled]
        self.t2_data = [self.t2_data[i] for i in inds_shuffled]

    def _extract_slices(self, data, slicenumbers, target_size=None):

        # initialize slices
        data_size = data[0].shape[1:]
        if target_size is None:
            slices = np.zeros((len(data), N_SLICES, data_size[0], data_size[1]), dtype='uint16')
        else:
            slices = np.zeros((len(data), N_SLICES, target_size[0], target_size[1]), dtype='uint16')

        # extract slices
        for i in range(len(data)):
            for j in range(N_SLICES):
                s = slicenumbers[i][j]
                if target_size is None:
                    slices[i, j, ...] = data[i][s]
                else:
                    # upsample to the target size
                    slices[i, j, ...] = cv2.resize(data[i][s], dsize=target_size, interpolation=cv2.INTER_CUBIC)

        return slices

    def _compute_clahe(self, data, clip_limit=T1_CLIPLIMIT):

        clahe = cv2.createCLAHE(clipLimit=clip_limit)
        data = data.astype(float)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                x = normalize(data[i, j, ...], type='minmax')
                data[i, j, ...] = clahe.apply((x * (2 ** 16 - 1)).astype('uint16')) / (2 ** 16 - 1)

        return data

    def _filter_bboxes(self, bboxes, sz):

        x_max, y_max = sz

        left_bboxes = []
        right_bboxes = []
        for bbox in bboxes:
            x, y, x_, y_ = bbox[:4]
            if x > x_max // 2 and x_ < x_max and y >= 0 and y_ < y_max:
                left_bboxes.append(bbox)
            elif x < x_max // 2 and x_ >= 0 and y >= 0 and y_ < y_max:
                right_bboxes.append(bbox)

        return np.asarray(left_bboxes), np.asarray(right_bboxes)

    def _maximize_bbox_score(self, bboxes):

        scores = bboxes[:, 4]
        i_max = np.argmax(scores)

        return bboxes[i_max]

    def _compute_joints(self, data, model, out_file=None):

        # load the joint model
        with set_layer_config(scriptable=False):
            bench = create_model(
                'tf_efficientdet_d0_mri',
                bench_task='predict',
                num_classes=1,
                redundant_bias=None,
                soft_nms=None,
                checkpoint_path=model,
            )
        bench = bench.cuda()
        amp_autocast = torch.cuda.amp.autocast
        bench.eval()

        # forward propagation
        bboxes = np.zeros((data.shape[0], N_SLICES, 2, 4))
        with torch.no_grad():
            for i in tqdm(range(data.shape[0]), desc='Computing SI joint locations'):
                x = data[i]
                x = (x - X_MU) / X_STD
                input = torch.from_numpy(np.repeat(x[:, np.newaxis, ...], 3, axis=1)).cuda()
                with amp_autocast():
                    output = bench(input.float()).cpu().numpy()
                for j in range(N_SLICES):
                    bb_l, bb_r = self._filter_bboxes(output[j], x.shape[1:])
                    bboxes[i, j, 0] = self._maximize_bbox_score(bb_l)[:4]
                    bboxes[i, j, 1] = self._maximize_bbox_score(bb_r)[:4]

        # save to tmp dir if necessary
        if out_file is not None:
            mkdir(os.path.dirname(out_file))
            np.save(out_file, bboxes)

        return bboxes

    def _compute_segmentation(self, data, model, out_file=None):

        # load the segmentation model
        net = UNet2D(feature_maps=FM, levels=LEVELS, norm=NORM, activation=ACTIVATION, coi=COI)
        net.load_state_dict(torch.load(model))
        net = net.cuda().eval()

        # forward propagation
        segmentation = np.zeros_like(data)
        with torch.no_grad():
            for i in tqdm(range(data.shape[0]), desc='Computing segmentation'):
                x = data[i]
                input = torch.from_numpy(x[:, np.newaxis, ...]).cuda()
                output = torch.softmax(net(input.float()), dim=1).cpu().numpy()
                segmentation[i, ...] = output[:, 1, ...]

        # save to tmp dir if necessary
        if out_file is not None:
            mkdir(os.path.dirname(out_file))
            np.save(out_file, segmentation)

        return segmentation

    def _synced_extraction(self, x_1, x_2, bbox, p_i, p_s, q_l, q_d, s):

        # bounding box coordinates
        i_ul, j_ul, i_br, j_br = bbox

        # joint location
        i_j = int((i_ul + i_br) // 2)
        j_j = int((j_ul + j_br) // 2)

        # joint angle
        alpha_j = np.arctan((i_br - i_ul) / (j_br - j_ul))

        # rotate image to fix joint region
        sgn = 1 if s == 0 else -1
        R = cv2.getRotationMatrix2D((i_j, j_j), (sgn * alpha_j) / np.pi * 180, 1)
        x_1_r = cv2.warpAffine(x_1, R, x_1.shape)
        x_2_r = cv2.warpAffine(x_2, R, x_2.shape)
        p_i_r = cv2.warpAffine(p_i, R, p_i.shape)
        p_s_r = cv2.warpAffine(p_s, R, p_s.shape)
        x_r = np.stack((x_1_r, x_2_r, p_i_r, p_s_r))

        # extract patches
        p = q_l + 2 * q_d
        if s == 0:
            ds = ((1, -1), (-1, -1), (-1, 1), (1, 1))
        else:
            ds = ((-1, -1), (1, -1), (1, 1), (-1, 1))
        q_ = np.zeros((x_r.shape[0], N_QUARTILES, p, p))
        for k, (di, dj) in enumerate(ds):
            i0, i1 = i_j-di*q_d, i_j+di*(q_l+q_d)
            j0, j1 = j_j-dj*q_d, j_j+dj*(q_l+q_d)
            i_start, i_stop = min(i0, i1), max(i0, i1)
            j_start, j_stop = min(j0, j1), max(j0, j1)
            q_[:, k, ...] = x_r[:, j_start:j_stop, i_start:i_stop]
        q = q_[:2, ...].astype('uint16')
        s = q_[2, ...]
        s[1:3] = q_[3, 1:3, ...]

        # compute normalized weights from segmentation
        w = self._compute_weights(s)

        return q, w

    def _extract_quartiles(self, t1_slices, t2_slices, si_joints, illium, sacrum, q_l, q_d, out_file=None):

        # allocate space for quartiles and corresponding weight maps
        n = len(t1_slices)
        p = q_l + 2 * q_d
        quartiles = np.zeros((n, 2, N_SLICES, N_SIDES, N_QUARTILES, p, p), dtype='uint16')
        weights = np.zeros((n, N_SLICES, N_SIDES, N_QUARTILES, p, p))

        # extract quartiles and weight maps
        for i in range(n):
            for j in range(N_SLICES):
                for k in range(N_SIDES):
                    q, w = self._synced_extraction(t1_slices[i, j], t2_slices[i, j], si_joints[i, j, k], illium[i, j],
                                                   sacrum[i, j], q_l, q_d, k)
                    quartiles[i, :, j, k] = q
                    weights[i, j, k] = w

        # save to tmp dir if necessary
        if out_file is not None:
            mkdir(os.path.dirname(out_file[0]))
            np.save(out_file[0], quartiles)
            np.save(out_file[1], weights)

        return quartiles, weights

    def _extract_scores(self, scores):

        n = len(scores[0])

        q_scores = np.asarray(scores[0], dtype=int)
        s_scores_i = np.asarray(scores[1], dtype=int)
        s_scores_d = np.asarray(scores[2], dtype=int)
        sparcc = np.sum(q_scores.reshape(n, -1), axis=1) + np.sum(s_scores_i.reshape(n, -1), axis=1) + \
                 np.sum(s_scores_i.reshape(n, -1), axis=1)

        return q_scores, s_scores_i, s_scores_d, sparcc


def _validate_shape(input_shape, data_shape, orientation=0, in_channels=1, levels=4):
    """
    Validates an input for propagation through a U-Net by taking into account the following:
        - Sampling along different orientations
        - Sampling multiple adjacent slices as channels
        - Maximum size that can be sampled from the data

    :param input_shape: original shape of the sample (Z, Y, X)
    :param data_shape: shape of the data to sample from (Z, Y, X)
    :param orientation: orientation to sample along
    :param in_channels: sample multiple adjacent slices as channels
    :param levels: amount of pooling layers in the network
    :return: the validated input shape
    """

    # make sure input shape can be edited
    input_shape = list(input_shape)

    # sample adjacent slices if necessary
    is2d = input_shape[0] == 1
    if is2d and in_channels > 1:
        input_shape[0] = in_channels

    # transform the data shape and input shape according to the orientation
    if orientation == 1:  # (Z, Y, X) -> (Y, Z, X)
        input_shape = [input_shape[1], input_shape[0], input_shape[2]]
    elif orientation == 2:  # (Z, Y, X) -> (X, Y, Z)
        input_shape = [input_shape[2], input_shape[1], input_shape[0]]

    # make sure the input shape fits in the data shape: i.e. find largest k such that n of the form n=k*2**levels
    for d in range(3):
        if not (is2d and d == orientation) and input_shape[d] > data_shape[d]:
            # 2D case: X, Y - 3D case: X, Y, Z
            # note we assume that the data has a least in_channels elements in each dimension
            input_shape[d] = int((data_shape[d] // (2 ** levels)) * (2 ** levels))

    # and return as a tuple
    return tuple(input_shape)


class VolumesDataset(data.Dataset):
    """
    Dataset for volumes

    :param data_path: path to the dataset
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional batch_size: size of the sampling batch
    :param optional dtype: type of the data (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional range_dir: orientation of the slicing
    """

    def __init__(self, data_path, input_shape, len_epoch=None, type='tif3d', in_channels=1, batch_size=1,
                 dtype='uint16', norm_type='unit', range_split=None, range_dir=None):
        self.data_path = data_path
        self.input_shape = input_shape
        self.len_epoch = len_epoch
        self.in_channels = in_channels
        self.orientation = 0
        self.batch_size = batch_size
        self.norm_type = norm_type
        self.range_split = range_split
        self.range_dir = range_dir

        # histogram equalizer
        clahe = cv2.createCLAHE(clipLimit=T1_CLIPLIMIT)

        # load the data
        self.data = []
        ls = os.listdir(data_path)
        ls.sort()
        for f in ls:
            data = read_volume(os.path.join(data_path, f), type=type, dtype=dtype)
            data = normalize(data.astype(float), type='minmax')
            for k in range(data.shape[0]):
                data[k] = clahe.apply((data[k] * (2 ** 16 - 1)).astype('uint16')) / (2 ** 16 - 1)
            self.data.append(data)
        self.tmp = self.data
        self.data = np.concatenate(self.data, axis=0)

        # select a subset of slices of the data
        self.data = slice_subset(self.data, range_split, range_dir)

        # compute length epoch if necessary
        if len_epoch is None or len_epoch < 0:
            print_frm('Epoch length not set... estimating full coverage value')
            t, self.len_epoch = _len_epoch(self.input_shape, self.data.shape)
            print_frm('Epoch length automatically set to %d, this covers %.2f%% of the data on average'
                      % (self.len_epoch, (1-t)*100))

    def __getitem__(self, i):
        pass

    def __len__(self):
        return self.len_epoch


class StronglyLabeledVolumesDataset(VolumesDataset):
    """
    Dataset for pixel-wise labeled volumes

    :param data_path: path to the dataset
    :param label_path: path to the labels
    :param input_shape: 3-tuple that specifies the input shape for sampling
    :param optional len_epoch: number of iterations for one epoch
    :param optional type: type of the volume file (tif2d, tif3d, tifseq, hdf5, png or pngseq)
    :param optional coi: list or sequence of the classes of interest
    :param optional in_channels: amount of subsequent slices to be sampled (only for 2D sampling)
    :param optional batch_size: size of the sampling batch
    :param optional data_dtype: type of the data (typically uint8)
    :param optional label_dtype: type of the labels (typically uint8)
    :param optional norm_type: type of the normalization (unit, z or minmax)
    :param optional transform: augmenter object
    :param optional range_split: range of slices (start, stop) to select (normalized between 0 and 1)
    :param optional range_dir: orientation of the slicing
    """

    def __init__(self, data_path, label_path, input_shape=None, len_epoch=None, type='tif3d', coi=(0, 1), in_channels=1,
                 batch_size=1, data_dtype='uint16', label_dtype='uint8', norm_type='unit', transform=None,
                 range_split=None, range_dir=None):
        super().__init__(data_path, input_shape, len_epoch=len_epoch, type=type, in_channels=in_channels,
                         batch_size=batch_size, dtype=data_dtype, norm_type=norm_type, range_split=range_split,
                         range_dir=range_dir)

        self.label_path = label_path
        self.coi = coi
        self.transform = transform
        if transform is not None:
            self.shared_transform, self.x_transform, self.y_transform = split_segmentation_transforms(transform)
        self.weight_balancing = None

        # load labels
        self.labels = []
        ls = os.listdir(label_path)
        ls.sort()
        for f in ls:
            labels = read_volume(os.path.join(label_path, f), type=type, dtype=label_dtype) - 1
            self.labels.append(labels)
        self.labels = np.concatenate(self.labels, axis=0)

        # select a subset of slices of the data
        self.labels = slice_subset(self.labels, range_split, range_dir)

    def __getitem__(self, i, attempt=0):

        # get shape of sample
        input_shape = _validate_shape(self.input_shape, self.data.shape, in_channels=self.in_channels,
                                      orientation=self.orientation)

        # get random sample
        x, y = sample_labeled_input(self.data, self.labels, input_shape)
        x = normalize(x, type=self.norm_type)
        y = y.astype(float)

        # add channel axis if the data is 3D
        if self.input_shape[0] > 1:
            x, y = x[np.newaxis, ...], y[np.newaxis, ...]

        # augment sample
        if self.transform is not None:
            data = self.shared_transform(np.concatenate((x, y), axis=0))
            p = x.shape[0]
            x = self.x_transform(data[:p])
            y = self.y_transform(data[p:])

        # transform to tensors
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).long()

        # return sample
        return x, y