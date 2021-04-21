'''
This script pre-processes the raw DICOM data and cleaned CSV annotations,
and converts them to a lossless compressed pickle format
'''

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

from neuralnets.util.io import print_frm

from util.constants import *
from util.tools import load


if __name__ == '__main__':

    # parse all the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to the directory that contains a preprocessed dataset", type=str, required=True)
    parser.add_argument("--print-scores", help="Flag that specifies whether to print the scores or not", action='store_true')
    args = parser.parse_args()

    # load the dataset
    scores = load(os.path.join(args.data_dir, SCORES_PP_FILE))
    sn = load(os.path.join(args.data_dir, SLICENUMBERS_PP_FILE))
    t1_data = load(os.path.join(args.data_dir, T1_PP_FILE))
    t2_data = load(os.path.join(args.data_dir, T2_PP_FILE))

    # select a random sample
    i = np.random.randint(len(t1_data))

    # visualize sample
    plt.figure(figsize=(12, 4), dpi=100)
    for k in range(6):
        plt.subplot(2, 6, k+1)
        plt.imshow(t1_data[i][sn[0][i][k]], cmap='gray')
        plt.axis('off')
        plt.subplot(2, 6, 6+k+1)
        plt.imshow(t2_data[i][sn[1][i][k]], cmap='gray')
        plt.axis('off')
    plt.show()

    # print scores if necessary
    print_frm('Amount of deficiency presences in indicated locations: ')
    types = ('Inflammation', 'Deep inflammation', 'Intense inflammation', 'Sclerosis', 'Erosion', 'Fat',
             'Partial Ankylosis', 'Ankylosis')
    sides = ['Left', 'Right']
    quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
    if args.print_scores:
        for k, t in enumerate(types):
            print_frm('%s: ' % t)
            if k == 1 or k == 2:
                for s in range(2):
                    print_frm('  %s: %s' % (sides[s], scores[k][i].sum(axis=0)[s]))
            else:
                for s in range(2):
                    for q in range(4):
                        print_frm('  %s - %s: %s' % (sides[s], quartiles[q], scores[k][i].sum(axis=0)[s, q]))
