'''
This file contains all constant values required in the repository.
'''

import numpy as np
import torchvision.models as models


# name of the CSV file containing the annotations
SCORES_FILE = 'scores-corrected.csv'
# suffix for the filtered data and merge directory name
SUFFIX_FILTERED = '-filtered-new'
MERGED_DIR = 'merged'

# preprocessed data files
SCORES_PP_FILE = 'scores.pickle'
DOMAINS_PP_FILE = 'domains.pickle'
SLICENUMBERS_PP_FILE = 'slicenumbers.pickle'
T1_PP_FILE = 't1.pickle'
T2_PP_FILE = 't2.pickle'

# different datasets
BEGIANT = 'BEGIANT'
HEALTHY_CONTROLS = 'HEALTHY_CONTROLS'
POPAS = 'POPAS'

# CSV fields from the raw data
PATIENT_NUMBER = "patientnumber"
ACCESSION_NUMBER = "Accessionnr"
IMAGE_ID = 'Image_ID'
READER = 'reader'

INFLAMMATION = 'Inflammation'
SCLEROSIS = 'Sclerosis'
EROSION = 'Erosion'
FAT = 'Fat'
PARTANK = 'PartAnk'
ANKYLOSIS = 'Ankylosis'
TYPES = [INFLAMMATION, SCLEROSIS, EROSION, FAT, PARTANK, ANKYLOSIS]

DEPTH = 'Depth'
INTENSITY = 'Intensity'

INFLAMMATORY = 'Inflammatory'
STRUCTURAL = 'Structural'
SLICENUMBER = 'slicenumber'

SLICE1 = 'slice1'
SLICE2 = 'slice2'
SLICE3 = 'slice3'
SLICE4 = 'slice4'
SLICE5 = 'slice5'
SLICE6 = 'slice6'
SIDE1 = 'L'
SIDE2 = 'R'
Q1 = 'Q1'
Q2 = 'Q2'
Q3 = 'Q3'
Q4 = 'Q4'
SLICES = [SLICE1, SLICE2, SLICE3, SLICE4, SLICE5, SLICE6]
SIDES = [SIDE1, SIDE2]
QUARTILES = [Q1, Q2, Q3, Q4]

T1 = 't1'
T2 = 't2'

T1_OPTIONS = {BEGIANT: ['t1_se_cor'],
              HEALTHY_CONTROLS: ['cor tse t1'],
              POPAS: ['t1_se_cor']}
T2_OPTIONS = {BEGIANT: ['t2_tirm_cor_320_3mm_pat2'],
              HEALTHY_CONTROLS: ['t2_tse_stir_cor_p2'],
              POPAS: ['t2_tirm_cor_320_3mm_pat2']}

T1_CLIPLIMIT = 200
T2_CLIPLIMIT = 400

i_ref = {T1: 5, T2: 5}
j_ref = {T1: 3, T2: 3}
Q_ALPHA = {T1: 70 / 180 * np.pi, T2: 70 / 180 * np.pi}

PADDING = 50

# X_MU = 0.3104745935741329
X_MU = 0.456
# X_STD = 0.2466064368858386
X_STD = 0.225

FM = 16
LEVELS = 4
NORM = 'batch'
ACTIVATION = 'relu'
COI = (0, 1)

N_SLICES = 6
N_SIDES = 2
N_QUARTILES = 4

Q_L = 64
Q_D = 10

CACHE_DIR = '/home/jorisro/research/mri-inflammation-prediction/.cache/mri'
SI_JOINTS_TMP = 'si_joints'
SEG_I_TMP = 'seg_i'
SEG_S_TMP = 'seg_s'
Q_TMP = 'quartiles'
W_TMP = 'weights'
EXT = 'npy'

ALPHA = 0.75

# network parameters
CONV_CHANNELS = [16, 32, 64, 128, 256]
FC_CHANNELS_INFLAMMATION = [64, 16, 2]
FC_CHANNELS_DEEP_INFLAMMATION = [64, 16, 2]
FC_CHANNELS_INTENSE_INFLAMMATION = [64, 16, 2]
KERNEL_SIZE = 3
CNN_NORM = 'batch'
CNN_ACTIVATION = 'relu'

# supported backbones
BACKBONES = {'AlexNet': models.alexnet,
             'VGG11': models.vgg11,
             'VGG16': models.vgg16,
             'ResNet18': models.resnet18,
             'ResNet101': models.resnet101,
             'ResNeXt101': models.resnext101_32x8d,
             'DenseNet121': models.densenet121,
             'DenseNet201': models.densenet201}

w_pos_inflamm = 0.082915473
w_neg_inflamm = 1 - w_pos_inflamm
INFLAMMATION_WEIGHTS = [1 / w_neg_inflamm, 1 / w_pos_inflamm]

w_pos_intense_inflamm = 0.038443171
w_neg_intense_inflamm = 1 - w_pos_inflamm
INTENSE_INFLAMMATION_WEIGHTS = [1 / w_neg_intense_inflamm, 1 / w_pos_intense_inflamm]

INFLAMMATION_MODULE = 'inflammation-module'
DEEP_INFLAMMATION_MODULE = 'deep-inflammation-module'
SPARCC_MODULE = 'sparcc-module'
JOINT = 'joint'

BINS = 50

REPS = 5

MEDIAN_THRESHOLD = 75

METRICS = ['accuracy', 'balanced-accuracy', 'precision', 'recall', 'fpr', 'f1-score', 'roc-auc', 'loss']

OPTIMAL_CKPT = 'best_model.ckpt'

MAX_ANGLE = 10