
import torch
import torch.utils.data as data
import os
import cv2
from neuralnets.util.tools import normalize, sample_labeled_input
from neuralnets.util.io import print_frm, read_volume
from neuralnets.data.base import slice_subset, _len_epoch
from neuralnets.data.datasets import split_segmentation_transforms

from util.constants import *


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