'''
This script pre-processes the raw DICOM data and cleaned CSV annotations,
and converts them to a lossless compressed pickle format
'''

import numpy as np
import pandas as pd
import pydicom
import os, shutil
import argparse

from tqdm import tqdm
from neuralnets.util.io import print_frm, mkdir

from util.constants import *
from util.tools import save, num2str


def _flatten_directory(d):
    sdirs = os.listdir(d)
    for sdir in sdirs:
        if os.path.isdir(os.path.join(d, sdir)):
            sdir = os.listdir(d)[0]
            pdir = os.listdir(os.path.join(d, sdir))[0]
            fs = os.listdir(os.path.join(d, sdir, pdir))
            for f in fs:
                fs_ = os.listdir(os.path.join(d, sdir, pdir, f))
                for f_ in fs_:
                    os.rename(os.path.join(d, sdir, pdir, f, f_), os.path.join(d, f_))
            shutil.rmtree(os.path.join(d, sdir))


def _is_t1(s, case, match=None):
    # for option in T1_OPTIONS[case]:
    #     if option in s:
    #         return True
    # return False
    if case == BEGIANT_VAL and match is not None:
        return s == match
    else:
        return s in T1_OPTIONS[case]


def _is_t2(s, case, match=None):
    # for option in T1_OPTIONS[case]:
    #     if option in s:
    #         return True
    # return False
    if case == BEGIANT_VAL and match is not None:
        return s == match
    else:
        return s in T2_OPTIONS[case]


def _extract_ids(sort_ids):
    ids = {}
    for t in [T1, T2]:
        # convert to integers
        s = np.asarray(sort_ids[t], dtype=int)
        # sort the indices
        i = np.argsort(s)
        # extract correct indices
        s_ = np.zeros_like(s)
        for j in range(len(s)):
            s_[i[j]] = j
        ids[t] = s_

    return ids


def _load_scores_begiant(scores_path):
    """
    Loads the scores CSV file and processes them in a structured format

    :param scores_path: path to the file
    :return: df_scores, (inflammation_L1_scores, inflammation_L21_scores, inflammation_L22_scores,
             sclerosis_L1_scores, erosion_L1_scores, fat_L1_scores, partankylosis_L1_scores,
             ankylosis_L1_scores), (slicenumbers_inflammation, slicenumbers_structural)
        df_scores: dataframe of the scores (median filtered over the different annotators)
        lists of scores of the following structure:
        - <x>_L1_scores: list of numpy arrays S (of shape [6, 2, 4]) such that S[i, j, k] is True if <x> is
                         present in slice i, side j, quartile k.
        - <x>_L21_scores: list of numpy arrays S (of shape [6, 2]) such that S[i, j] is True if deep <x> lesion
                          is present in slice i, side j.
        - <x>_L22_scores: list of numpy arrays S (of shape [6, 2]) such that S[i, j] is True if intense <x>
                          lesion is present in slice i, side j.
        lists of slicenumbers:
        - slicenumbers_inflammation: list of slicenumbers that were selected for inflammation scoring
        - slicenumbers_structural: list of slicenumbers that were selected for structural scoring

    """
    # load scores dataframe
    df = pd.read_csv(scores_path, sep=',')

    # sort w.r.t. patient numbers
    df = df.sort_values(PATIENT_NUMBER)

    # for each patient and study, compute the median scores across the different specialists
    df_groups = df.groupby([PATIENT_NUMBER, ACCESSION_NUMBER])
    df_scores = df_groups.median()
    # df_scores = df_scores.dropna(how="any").copy()
    # slicenumbers should be integers
    for type in [INFLAMMATORY, STRUCTURAL]:
        for slice in SLICES:
            col = type + '_' + SLICENUMBER + '_' + slice
            df_scores[col] = df_scores[col].astype(int)

    # extract quartile scores
    L1_scores = {INFLAMMATION: [],
                 SCLEROSIS: [],
                 EROSION: [],
                 FAT: [],
                 PARTANK: [],
                 ANKYLOSIS: []}
    slicenumbers_inflammation = []
    slicenumbers_structural = []
    for id, score in df_scores.iterrows():
        si = np.zeros((len(SLICES)), dtype=int)
        ss = np.zeros((len(SLICES)), dtype=int)
        for dis_type in TYPES:
            q_scores = np.zeros((len(SLICES), len(SIDES), len(QUARTILES)), dtype=bool)
            for i, slice in enumerate(SLICES):
                si[i] = getattr(score, INFLAMMATORY + '_' + SLICENUMBER + '_' + slice) - 1
                ss[i] = getattr(score, STRUCTURAL + '_' + SLICENUMBER + '_' + slice) - 1
                for j, side in enumerate(SIDES):
                    for k, quartile in enumerate(QUARTILES):
                        q_scores[i, j, k] = \
                            getattr(score, dis_type + '_' + quartile + '_' + side + '_' + slice)
            L1_scores[dis_type].append(q_scores)
        slicenumbers_inflammation.append(si)
        slicenumbers_structural.append(ss)

    # extract side scores
    inflammation_L21_scores = []
    inflammation_L22_scores = []
    for id, score in df_scores.iterrows():
        s_scores_1 = np.zeros((len(SLICES), len(SIDES)), dtype=bool)
        s_scores_2 = np.zeros((len(SLICES), len(SIDES)), dtype=bool)
        for i, slice in enumerate(SLICES):
            for j, side in enumerate(SIDES):
                s_scores_1[i, j] = getattr(score, DEPTH + '_' + side + '_' + slice)
                s_scores_1[i, j] = getattr(score, INTENSITY + '_' + side + '_' + slice)
        inflammation_L21_scores.append(s_scores_1)
        inflammation_L22_scores.append(s_scores_2)

    return df_scores, \
           (L1_scores[INFLAMMATION], inflammation_L21_scores, inflammation_L22_scores, L1_scores[SCLEROSIS], \
            L1_scores[EROSION], L1_scores[FAT], L1_scores[PARTANK], L1_scores[ANKYLOSIS]), \
           (slicenumbers_inflammation, slicenumbers_structural)


def _load_scores_begiant_val(scores_path):
    """
    Loads the scores CSV file and processes them in a structured format

    :param scores_path: path to the file
    :return: df_scores, (inflammation_L1_scores, inflammation_L21_scores, inflammation_L22_scores,
             sclerosis_L1_scores, erosion_L1_scores, fat_L1_scores, partankylosis_L1_scores,
             ankylosis_L1_scores), (slicenumbers_inflammation, slicenumbers_structural)
        df_scores: dataframe of the scores (median filtered over the different annotators)
        lists of scores of the following structure:
        - <x>_L1_scores: list of numpy arrays S (of shape [6, 2, 4]) such that S[i, j, k] is True if <x> is
                         present in slice i, side j, quartile k.
        - <x>_L21_scores: list of numpy arrays S (of shape [6, 2]) such that S[i, j] is True if deep <x> lesion
                          is present in slice i, side j.
        - <x>_L22_scores: list of numpy arrays S (of shape [6, 2]) such that S[i, j] is True if intense <x>
                          lesion is present in slice i, side j.
        lists of slicenumbers:
        - slicenumbers_inflammation: list of slicenumbers that were selected for inflammation scoring
        - slicenumbers_structural: list of slicenumbers that were selected for structural scoring

    """
    # load scores dataframe
    df = pd.read_csv(scores_path, sep=',')

    # sort w.r.t. patient numbers
    df = df.sort_values(PATIENT_NUMBER)

    # for each patient and study, compute the median scores across the different specialists
    dt1 = df[[PATIENT_NUMBER, ACCESSION_NUMBER, DESCRIPTION_T1]]
    dt1 = dt1.groupby([PATIENT_NUMBER, ACCESSION_NUMBER]).max()
    dt2 = df[[PATIENT_NUMBER, ACCESSION_NUMBER, DESCRIPTION_T2]]
    dt2 = dt2.groupby([PATIENT_NUMBER, ACCESSION_NUMBER]).max()
    df = df.drop(columns=[DESCRIPTION_T1, DESCRIPTION_T2])
    df_groups = df.groupby([PATIENT_NUMBER, ACCESSION_NUMBER])
    df_scores = df_groups.median()
    df_scores = pd.concat([df_scores, dt1, dt2], axis=1)
    # df_scores = df_scores.dropna(how="any").copy()
    # slicenumbers should be integers
    for type in [INFLAMMATORY, STRUCTURAL]:
        for slice in SLICES:
            col = type + '_' + SLICENUMBER + '_' + slice
            df_scores[col] = df_scores[col].astype(int)

    # extract quartile scores
    L1_scores = {INFLAMMATION: [],
                 SCLEROSIS: [],
                 EROSION: [],
                 FAT: [],
                 PARTANK: [],
                 ANKYLOSIS: []}
    slicenumbers_inflammation = []
    slicenumbers_structural = []
    for id, score in df_scores.iterrows():
        si = np.zeros((len(SLICES)), dtype=int)
        ss = np.zeros((len(SLICES)), dtype=int)
        for dis_type in TYPES:
            q_scores = np.zeros((len(SLICES), len(SIDES), len(QUARTILES)), dtype=bool)
            for i, slice in enumerate(SLICES):
                si[i] = getattr(score, INFLAMMATORY + '_' + SLICENUMBER + '_' + slice) - 1
                ss[i] = getattr(score, STRUCTURAL + '_' + SLICENUMBER + '_' + slice) - 1
                for j, side in enumerate(SIDES):
                    for k, quartile in enumerate(QUARTILES):
                        q_scores[i, j, k] = \
                            getattr(score, dis_type + '_' + quartile + '_' + side + '_' + slice)
            L1_scores[dis_type].append(q_scores)
        slicenumbers_inflammation.append(si)
        slicenumbers_structural.append(ss)

    # extract side scores
    inflammation_L21_scores = []
    inflammation_L22_scores = []
    for id, score in df_scores.iterrows():
        s_scores_1 = np.zeros((len(SLICES), len(SIDES)), dtype=bool)
        s_scores_2 = np.zeros((len(SLICES), len(SIDES)), dtype=bool)
        for i, slice in enumerate(SLICES):
            for j, side in enumerate(SIDES):
                s_scores_1[i, j] = getattr(score, DEPTH + '_' + side + '_' + slice)
                s_scores_1[i, j] = getattr(score, INTENSITY + '_' + side + '_' + slice)
        inflammation_L21_scores.append(s_scores_1)
        inflammation_L22_scores.append(s_scores_2)

    return df_scores, \
           (L1_scores[INFLAMMATION], inflammation_L21_scores, inflammation_L22_scores, L1_scores[SCLEROSIS], \
            L1_scores[EROSION], L1_scores[FAT], L1_scores[PARTANK], L1_scores[ANKYLOSIS]), \
           (slicenumbers_inflammation, slicenumbers_structural)


def _load_scores_healthy(scores_path):
    """
    Loads the scores CSV file and processes them in a structured format

    :param scores_path: path to the file
    :return: df_scores, (inflammation_L1_scores, inflammation_L21_scores, inflammation_L22_scores,
             sclerosis_L1_scores, erosion_L1_scores, fat_L1_scores, partankylosis_L1_scores,
             ankylosis_L1_scores), (slicenumbers_inflammation, slicenumbers_structural)
        df_scores: dataframe of the scores (median filtered over the different annotators)
        lists of scores of the following structure:
        - <x>_L1_scores: list of numpy arrays S (of shape [6, 2, 4]) such that S[i, j, k] is True if <x> is
                         present in slice i, side j, quartile k.
        - <x>_L21_scores: list of numpy arrays S (of shape [6, 2]) such that S[i, j] is True if deep <x> lesion
                          is present in slice i, side j.
        - <x>_L22_scores: list of numpy arrays S (of shape [6, 2]) such that S[i, j] is True if intense <x>
                          lesion is present in slice i, side j.
        lists of slicenumbers:
        - slicenumbers_inflammation: list of slicenumbers that were selected for inflammation scoring
        - slicenumbers_structural: list of slicenumbers that were selected for structural scoring

    """
    # load scores dataframe
    df = pd.read_csv(scores_path, sep=',')

    # split dataframe in three parts along columns:
    #   - TR annotations
    #   - MdH annotations
    #   - Remaining ones
    cols_ = []
    for c in list(df.columns):
        if '_MdH' in c:
            cols_.append(c)
    df_ = df[cols_]
    df_.columns = [c[:-4] for c in list(df_.columns)]
    df_scores = pd.concat((df_, df[IMAGE_ID]), axis=1)

    # extract quartile scores
    L1_scores = {INFLAMMATION: [],
                 SCLEROSIS: [],
                 EROSION: [],
                 FAT: [],
                 PARTANK: [],
                 ANKYLOSIS: []}
    slicenumbers_inflammation = []
    slicenumbers_structural = []
    for id, score in df_scores.iterrows():
        si = np.zeros((len(SLICES)), dtype=int)
        ss = np.zeros((len(SLICES)), dtype=int)
        for dis_type in TYPES:
            q_scores = np.zeros((len(SLICES), len(SIDES), len(QUARTILES)), dtype=bool)
            for i, slice in enumerate(SLICES):
                si[i] = getattr(score, INFLAMMATORY + '_' + SLICENUMBER + '_' + slice) - 1
                ss[i] = getattr(score, STRUCTURAL + '_' + SLICENUMBER + '_' + slice) - 1
                for j, side in enumerate(SIDES):
                    for k, quartile in enumerate(QUARTILES):
                        q_scores[i, j, k] = \
                            getattr(score, dis_type + '_' + quartile + '_' + side + '_' + slice)
            L1_scores[dis_type].append(q_scores)
        slicenumbers_inflammation.append(si)
        slicenumbers_structural.append(ss)

    # extract side scores
    inflammation_L21_scores = []
    inflammation_L22_scores = []
    for id, score in df_scores.iterrows():
        s_scores_1 = np.zeros((len(SLICES), len(SIDES)), dtype=bool)
        s_scores_2 = np.zeros((len(SLICES), len(SIDES)), dtype=bool)
        for i, slice in enumerate(SLICES):
            for j, side in enumerate(SIDES):
                s_scores_1[i, j] = getattr(score, DEPTH + '_' + side + '_' + slice)
                s_scores_1[i, j] = getattr(score, INTENSITY + '_' + side + '_' + slice)
        inflammation_L21_scores.append(s_scores_1)
        inflammation_L22_scores.append(s_scores_2)

    return df_scores, \
           (L1_scores[INFLAMMATION], inflammation_L21_scores, inflammation_L22_scores, L1_scores[SCLEROSIS], \
            L1_scores[EROSION], L1_scores[FAT], L1_scores[PARTANK], L1_scores[ANKYLOSIS]), \
           (slicenumbers_inflammation, slicenumbers_structural)


def _print_stats_begiant(df_scores, scores):
    print_frm("===========")
    print_frm("Score stats")
    print_frm("===========")

    # count the number of patients
    n_patients = df_scores[PATIENT_NUMBER].nunique()
    print_frm("Found %d patients" % (n_patients))

    # number of MRI studies
    n_studies = df_scores.shape[0]
    print_frm("Found %d studies" % (n_studies))

    # number of L1 and L2 labels
    n_quartiles = n_studies * len(SLICES) * len(SIDES) * len(QUARTILES)
    n_sides = n_studies * len(SLICES) * len(SIDES)
    print_frm("Found %d L1 labels (for inflammation, sclerosis, erosion, fat, partial ankylosis and ankylosis)"
              % (n_quartiles))
    print_frm("Found %d L2 labels (for inflammation)" % (n_sides))

    # describe distribution of the L1 labels
    for i, j in enumerate([0, 3, 4, 5, 6, 7]):
        n_pos = 0
        for score in scores[j]:
            n_pos = n_pos + score.sum()
        print_frm("    %s L1 label distribution: %d (%.2f%%) positive, %d (%.2f%%) negative" %
                  (TYPES[i], n_pos, 100 * n_pos / n_quartiles, n_quartiles - n_pos, 100 * (1 - n_pos / n_quartiles)))

    # describe distribution of the L2 labels
    ALT_TYPES = [DEPTH, INTENSITY]
    for i, j in enumerate([1, 2]):
        n_pos = 0
        for score in scores[j]:
            n_pos = n_pos + score.sum()
        print_frm("    %s L2 label distribution: %d (%.2f%%) positive, %d (%.2f%%) negative" %
                  (ALT_TYPES[i], n_pos, 100 * n_pos / n_sides, n_sides - n_pos, 100 * (1 - n_pos / n_sides)))

    print_frm("===========")


def _print_stats_healthy(df_scores, scores):
    print_frm("===========")
    print_frm("Score stats")
    print_frm("===========")

    # count the number of patients
    n_patients = df_scores[IMAGE_ID].nunique()
    print_frm("Found %d patients" % (n_patients))

    # number of MRI studies
    n_studies = df_scores.shape[0]
    print_frm("Found %d studies" % (n_studies))

    # number of L1 and L2 labels
    n_quartiles = n_studies * len(SLICES) * len(SIDES) * len(QUARTILES)
    n_sides = n_studies * len(SLICES) * len(SIDES)
    print_frm("Found %d L1 labels (for inflammation, sclerosis, erosion, fat, partial ankylosis and ankylosis)"
              % (n_quartiles))
    print_frm("Found %d L2 labels (for inflammation)" % (n_sides))

    # describe distribution of the L1 labels
    for i, j in enumerate([0, 3, 4, 5, 6, 7]):
        n_pos = 0
        for score in scores[j]:
            n_pos = n_pos + score.sum()
        print_frm("    %s L1 label distribution: %d (%.2f%%) positive, %d (%.2f%%) negative" %
                  (TYPES[i], n_pos, 100 * n_pos / n_quartiles, n_quartiles - n_pos, 100 * (1 - n_pos / n_quartiles)))

    # describe distribution of the L2 labels
    ALT_TYPES = [DEPTH, INTENSITY]
    for i, j in enumerate([1, 2]):
        n_pos = 0
        for score in scores[j]:
            n_pos = n_pos + score.sum()
        print_frm("    %s L2 label distribution: %d (%.2f%%) positive, %d (%.2f%%) negative" %
                  (ALT_TYPES[i], n_pos, 100 * n_pos / n_sides, n_sides - n_pos, 100 * (1 - n_pos / n_sides)))

    print_frm("===========")


def _filter_relevant_begiant(data_path, df_scores, scores, slicenumbers):
    """
    Loads the data that corresponds to the selected score files and filters the relevant ones out

    :param data_path: path the data that contain the patient directories
    :param df_scores: scores data frame
    :param scores: preloaded scores
    :param slicenumbers: slicenumbers that were scored for each patient study
    :return: Data that corresponds to the selected score files and corrected score files (if this was necessary)
        (t1_images, t2_images), (df_scores, scores, slicenumbers)
        - t1_images: list of T1 images that belong to the corresponding structural scores
        - t2_images: list of T2 images that belong to the corresponding inflammation scores
        - df_scores: filtered scores data frame (records may be removed due to absence of T1/T2 image)
        - scores: filtered preloaded scores (records may be removed due to absence of T1/T2 image)
        - slicenumbers: filtered slicenumbers (records may be removed due to absence of T1/T2 image)
    """
    ds = os.path.basename(data_path)
    # loop over all scored samples
    df_scores = df_scores.reset_index()
    filtered_df_scores = pd.DataFrame(columns=df_scores.columns)
    filtered_scores = [[] for j in range(len(scores))]
    filtered_slicenumbers = [[] for j in range(len(slicenumbers))]
    filtered_volumes = {T1: [], T2: []}
    irows = list(df_scores.iterrows())
    for k in tqdm(range(len(irows))):
        i, p_rec = irows[k]
        p_id = p_rec[PATIENT_NUMBER]
        p_acn = p_rec[ACCESSION_NUMBER]

        # initialize data list
        volumes_tmp = {T1: [], T2: []}
        sort_ids = {T1: [], T2: []}

        # loop over all sequences and filter relevant ones
        pt_dir = 'PAT' + num2str(p_id, K=4)
        patient_path = os.path.join(data_path, pt_dir, 'MR', pt_dir + '_' + p_acn)
        if not os.path.isdir(patient_path):
            patient_path = os.path.join(data_path, pt_dir, 'MR', 'PAT' + '0' + str(p_id) + '_' + p_acn)
            if not os.path.isdir(patient_path):
                # print('Non existing directory: %s' % patient_path)
                continue

        image_files = os.listdir(patient_path)
        if os.path.isdir(os.path.join(patient_path, image_files[0])):
            _flatten_directory(patient_path)
            continue
        x = []
        for image_file in image_files:
            # read the image
            image = pydicom.read_file(os.path.join(patient_path, image_file))
            if image.SeriesDescription not in x:
                x.append(image.SeriesDescription)
            if _is_t1(image.SeriesDescription, ds):
                # print_frm('   Found T1 image: %s' % (image_file))
                volumes_tmp[T1].append(image.pixel_array)
                sort_ids[T1].append(image.InstanceNumber)
            elif _is_t2(image.SeriesDescription, ds):
                # print_frm('   Found T2 image: %s' % (image_file))
                volumes_tmp[T2].append(image.pixel_array)
                sort_ids[T2].append(image.InstanceNumber)

        # extract indices from sorted ids
        sort_ids = _extract_ids(sort_ids)

        # check None values
        keep_sample = True
        for id in [T1, T2]:
            if len(volumes_tmp[id]) == 0:
                # print_frm('    Discarding patient %d, study %s due to non-existing %s data' % (p_id, p_acn, id))
                keep_sample = False
                print(os.path.basename(patient_path))

        # sort images w.r.t. instance numbers
        if keep_sample:
            for id in [T1, T2]:
                n_images = len(volumes_tmp[id])
                shape = volumes_tmp[id][0].shape
                dtype = volumes_tmp[id][0].dtype
                volume = np.zeros((n_images, *shape), dtype=dtype)
                for n in range(n_images):
                    volume[sort_ids[id][n]] = volumes_tmp[id][n]
                filtered_volumes[id].append(volume)

            # perform filtering, i.e. save record
            filtered_df_scores = filtered_df_scores.append(p_rec)
            for j in range(len(filtered_scores)):
                filtered_scores[j].append(scores[j][i])
            for j in range(len(filtered_slicenumbers)):
                filtered_slicenumbers[j].append(slicenumbers[j][i])

    return (filtered_volumes[T1], filtered_volumes[T2]), \
           (filtered_df_scores, tuple(filtered_scores), tuple(filtered_slicenumbers))


def _filter_relevant_begiant_val(data_path, df_scores, scores, slicenumbers):
    """
    Loads the data that corresponds to the selected score files and filters the relevant ones out

    :param data_path: path the data that contain the patient directories
    :param df_scores: scores data frame
    :param scores: preloaded scores
    :param slicenumbers: slicenumbers that were scored for each patient study
    :return: Data that corresponds to the selected score files and corrected score files (if this was necessary)
        (t1_images, t2_images), (df_scores, scores, slicenumbers)
        - t1_images: list of T1 images that belong to the corresponding structural scores
        - t2_images: list of T2 images that belong to the corresponding inflammation scores
        - df_scores: filtered scores data frame (records may be removed due to absence of T1/T2 image)
        - scores: filtered preloaded scores (records may be removed due to absence of T1/T2 image)
        - slicenumbers: filtered slicenumbers (records may be removed due to absence of T1/T2 image)
    """
    ds = os.path.basename(data_path)
    # loop over all scored samples
    df_scores = df_scores.reset_index()
    filtered_df_scores = pd.DataFrame(columns=df_scores.columns)
    filtered_scores = [[] for j in range(len(scores))]
    filtered_slicenumbers = [[] for j in range(len(slicenumbers))]
    filtered_volumes = {T1: [], T2: []}
    irows = list(df_scores.iterrows())
    for k in tqdm(range(len(irows))):
        i, p_rec = irows[k]
        p_id = p_rec[PATIENT_NUMBER]
        p_acn = str(p_rec[ACCESSION_NUMBER])

        # initialize data list
        volumes_tmp = {T1: [], T2: []}
        sort_ids = {T1: [], T2: []}

        # loop over all sequences and filter relevant ones
        pt_dir = 'PAT' + num2str(p_id, K=4)
        patient_path = os.path.join(data_path, pt_dir, 'MR', pt_dir + '_' + p_acn)
        if not os.path.isdir(patient_path):
            patient_path = os.path.join(data_path, pt_dir, 'MR', 'PAT' + '0' + str(p_id) + '_' + p_acn)
            if not os.path.isdir(patient_path):
                p_acn = str(int(p_acn) - 1)
                patient_path = os.path.join(data_path, pt_dir, 'MR', 'PAT' + '0' + str(p_id) + '_' + p_acn)
                if not os.path.isdir(patient_path):
                    p_acn = str(int(p_acn) + 2)
                    patient_path = os.path.join(data_path, pt_dir, 'MR', 'PAT' + '0' + str(p_id) + '_' + p_acn)
                    if not os.path.isdir(patient_path):
                        p_acn = str(int(p_acn) - 1)
                        patient_path = os.path.join(pt_dir, 'MR', 'PAT' + '0' + str(p_id) + '_' + p_acn)
                        # print_frm('Non existing directory: %s' % patient_path)
                        continue

        image_files = os.listdir(patient_path)
        if os.path.isdir(os.path.join(patient_path, image_files[0])):
            _flatten_directory(patient_path)
            continue
        x = []
        for image_file in image_files:
            # read the image
            image = pydicom.read_file(os.path.join(patient_path, image_file))
            if image.SeriesDescription not in x:
                x.append(image.SeriesDescription)
            if _is_t1(image.SeriesDescription, ds, match=p_rec[DESCRIPTION_T1]):
                # print_frm('   Found T1 image: %s' % (image_file))
                volumes_tmp[T1].append(image.pixel_array)
                sort_ids[T1].append(image.InstanceNumber)
            elif _is_t2(image.SeriesDescription, ds, match=p_rec[DESCRIPTION_T2]):
                # print_frm('   Found T2 image: %s' % (image_file))
                volumes_tmp[T2].append(image.pixel_array)
                sort_ids[T2].append(image.InstanceNumber)

        # extract indices from sorted ids
        sort_ids = _extract_ids(sort_ids)

        # check None values
        keep_sample = True
        for id in [T1, T2]:
            if len(volumes_tmp[id]) == 0:
                # print_frm('    Discarding patient %d, study %s due to non-existing %s data' % (p_id, p_acn, id))
                keep_sample = False
                patient_path = os.path.join(pt_dir, 'MR', 'PAT' + '0' + str(p_id) + '_' + p_acn)
                # print_frm(f'Missing {id.upper()} sequence in {patient_path}')
                print_frm(x)

        # sort images w.r.t. instance numbers
        if keep_sample:
            for id in [T1, T2]:
                n_images = len(volumes_tmp[id])
                shape = volumes_tmp[id][0].shape
                dtype = volumes_tmp[id][0].dtype
                volume = np.zeros((n_images, *shape), dtype=dtype)
                for n in range(n_images):
                    volume[sort_ids[id][n]] = volumes_tmp[id][n]
                filtered_volumes[id].append(volume)

            # perform filtering, i.e. save record
            filtered_df_scores = filtered_df_scores.append(p_rec)
            for j in range(len(filtered_scores)):
                filtered_scores[j].append(scores[j][i])
            for j in range(len(filtered_slicenumbers)):
                filtered_slicenumbers[j].append(slicenumbers[j][i])

    return (filtered_volumes[T1], filtered_volumes[T2]), \
           (filtered_df_scores, tuple(filtered_scores), tuple(filtered_slicenumbers))


def _filter_relevant_healthy(data_path, df_scores, scores, slicenumbers):
    """
    Loads the data that corresponds to the selected score files and filters the relevant ones out

    :param data_path: path the data that contain the patient directories
    :param df_scores: scores data frame
    :param scores: preloaded scores
    :param slicenumbers: slicenumbers that were scored for each patient study
    :return: Data that corresponds to the selected score files and corrected score files (if this was necessary)
        (t1_images, t2_images), (df_scores, scores, slicenumbers)
        - t1_images: list of T1 images that belong to the corresponding structural scores
        - t2_images: list of T2 images that belong to the corresponding inflammation scores
        - df_scores: filtered scores data frame (records may be removed due to absence of T1/T2 image)
        - scores: filtered preloaded scores (records may be removed due to absence of T1/T2 image)
        - slicenumbers: filtered slicenumbers (records may be removed due to absence of T1/T2 image)
    """
    ds = os.path.basename(data_path)
    # loop over all scored samples
    df_scores = df_scores.reset_index()
    filtered_df_scores = pd.DataFrame(columns=df_scores.columns)
    filtered_scores = [[] for j in range(len(scores))]
    filtered_slicenumbers = [[] for j in range(len(slicenumbers))]
    filtered_volumes = {T1: [], T2: []}
    irows = list(df_scores.iterrows())
    for k in tqdm(range(len(irows))):
        i, p_rec = irows[k]
        p_id, p_st = p_rec[IMAGE_ID].split('_')
        p_id_int = int(p_id)
        p_id_str = num2str(p_id_int, K=4)
        # print_frm('Loading patient %d/%d - study %s' % (p_id_int, int(p_max_id[-2:]), p_st))

        # initialize data list
        volumes_tmp = {T1: [], T2: []}
        sort_ids = {T1: [], T2: []}

        # loop over all sequences and filter relevant ones
        pt_dir = 'PAT' + p_id_str
        patient_path = os.path.join(data_path, pt_dir, 'Study ' + p_st)
        series_dirs = os.listdir(patient_path)
        series_dirs.sort()
        for series_dir in series_dirs:
            # read the image
            image_files = os.listdir(os.path.join(patient_path, series_dir))
            image_files.sort()
            for image_file in image_files:
                image = pydicom.read_file(os.path.join(patient_path, series_dir, image_file))
                if _is_t1(image.SeriesDescription, ds):
                    volumes_tmp[T1].append(image.pixel_array)
                    sort_ids[T1].append(image.InstanceNumber)
                elif _is_t2(image.SeriesDescription, ds):
                    volumes_tmp[T2].append(image.pixel_array)
                    sort_ids[T2].append(image.InstanceNumber)

        # extract indices from sorted ids
        sort_ids = _extract_ids(sort_ids)

        # check None values
        keep_sample = True
        for id in [T1, T2]:
            if len(volumes_tmp[id]) == 0:
                # print_frm('    Discarding patient %d, study %s due to non-existing %s data' % (p_id_int, p_st, id))
                keep_sample = False

        # sort images w.r.t. instance numbers
        if keep_sample:
            for id in [T1, T2]:
                n_images = len(volumes_tmp[id])
                shape = volumes_tmp[id][0].shape
                dtype = volumes_tmp[id][0].dtype
                volume = np.zeros((n_images, *shape), dtype=dtype)
                for n in range(n_images):
                    volume[sort_ids[id][n]] = volumes_tmp[id][n]
                filtered_volumes[id].append(volume)

            # perform filtering, i.e. save record
            filtered_df_scores = filtered_df_scores.append(p_rec)
            for j in range(len(filtered_scores)):
                filtered_scores[j].append(scores[j][i])
            for j in range(len(filtered_slicenumbers)):
                filtered_slicenumbers[j].append(slicenumbers[j][i])

    return (filtered_volumes[T1], filtered_volumes[T2]), \
           (filtered_df_scores, tuple(filtered_scores), tuple(filtered_slicenumbers))


def _filter_relevant_popas(data_path, df_scores, scores, slicenumbers):
    """
    Loads the data that corresponds to the selected score files and filters the relevant ones out

    :param data_path: path the data that contain the patient directories
    :param df_scores: scores data frame
    :param scores: preloaded scores
    :param slicenumbers: slicenumbers that were scored for each patient study
    :return: Data that corresponds to the selected score files and corrected score files (if this was necessary)
        (t1_images, t2_images), (df_scores, scores, slicenumbers)
        - t1_images: list of T1 images that belong to the corresponding structural scores
        - t2_images: list of T2 images that belong to the corresponding inflammation scores
        - df_scores: filtered scores data frame (records may be removed due to absence of T1/T2 image)
        - scores: filtered preloaded scores (records may be removed due to absence of T1/T2 image)
        - slicenumbers: filtered slicenumbers (records may be removed due to absence of T1/T2 image)
    """
    ds = os.path.basename(data_path)
    # loop over all scored samples
    df_scores = df_scores.reset_index()
    filtered_df_scores = pd.DataFrame(columns=df_scores.columns)
    filtered_scores = [[] for j in range(len(scores))]
    filtered_slicenumbers = [[] for j in range(len(slicenumbers))]
    filtered_volumes = {T1: [], T2: []}
    irows = list(df_scores.iterrows())
    for k in tqdm(range(len(irows))):
        i, p_rec = irows[k]
        p_id = p_rec[PATIENT_NUMBER]
        p_acn = p_rec[ACCESSION_NUMBER]
        p_id_int = int(p_id[-2:])
        p_id_str = num2str(p_id_int, K=3)
        # print_frm('Loading patient %d/%d - study %s' % (p_id_int, int(p_max_id[-2:]), p_acn))

        # initialize data list
        volumes_tmp = {T1: [], T2: []}
        sort_ids = {T1: [], T2: []}

        # loop over all sequences and filter relevant ones
        pt_dir = p_id[:2] + p_id_str + '_' + p_acn
        patient_path = os.path.join(data_path, pt_dir)
        image_files = os.listdir(patient_path)
        for image_file in image_files:
            # read the image
            image = pydicom.read_file(os.path.join(patient_path, image_file))
            if _is_t1(image.SeriesDescription, ds):
                # print_frm('   Found T1 image: %s' % (image_file))
                volumes_tmp[T1].append(image.pixel_array)
                sort_ids[T1].append(image.InstanceNumber)
            elif _is_t2(image.SeriesDescription, ds):
                # print_frm('   Found T2 image: %s' % (image_file))
                volumes_tmp[T2].append(image.pixel_array)
                sort_ids[T2].append(image.InstanceNumber)

        # extract indices from sorted ids
        sort_ids = _extract_ids(sort_ids)

        # check None values
        keep_sample = True
        for id in [T1, T2]:
            if len(volumes_tmp[id]) == 0:
                # print_frm('    Discarding patient %d, study %s due to non-existing %s data' % (p_id_int, p_acn, id))
                keep_sample = False

        # sort images w.r.t. instance numbers
        if keep_sample:
            for id in [T1, T2]:
                n_images = len(volumes_tmp[id])
                shape = volumes_tmp[id][0].shape
                dtype = volumes_tmp[id][0].dtype
                volume = np.zeros((n_images, *shape), dtype=dtype)
                for n in range(n_images):
                    volume[sort_ids[id][n]] = volumes_tmp[id][n]
                filtered_volumes[id].append(volume)

            # perform filtering, i.e. save record
            filtered_df_scores = filtered_df_scores.append(p_rec)
            for j in range(len(filtered_scores)):
                filtered_scores[j].append(scores[j][i])
            for j in range(len(filtered_slicenumbers)):
                filtered_slicenumbers[j].append(slicenumbers[j][i])

    return (filtered_volumes[T1], filtered_volumes[T2]), \
           (filtered_df_scores, tuple(filtered_scores), tuple(filtered_slicenumbers))


load_scores_fns = {BEGIANT: _load_scores_begiant,
                   BEGIANT_VAL: _load_scores_begiant_val,
                   POPAS: _load_scores_begiant,
                   HEALTHY_CONTROLS: _load_scores_healthy}
print_stats_fns = {BEGIANT: _print_stats_begiant,
                   BEGIANT_VAL: _print_stats_begiant,
                   POPAS: _print_stats_begiant,
                   HEALTHY_CONTROLS: _print_stats_healthy}
filter_relevant_fns = {BEGIANT: _filter_relevant_begiant,
                       BEGIANT_VAL: _filter_relevant_begiant_val,
                       POPAS: _filter_relevant_popas,
                       HEALTHY_CONTROLS: _filter_relevant_healthy}


if __name__ == '__main__':

    # parse all the arguments
    print_frm('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", help="Path to the directory that contains all data", type=str, required=True)
    parser.add_argument("--datasets", help="Subdirectories of datasets that should be processed", type=str, default='')
    parser.add_argument("--merge", help="Flag that specifies whether to merge the data or not", action='store_true')
    args = parser.parse_args()
    args.datasets = args.datasets.split(',')

    all_scores = []
    all_slicenumbers = []
    all_domains = []
    all_t1_volumes = []
    all_t2_volumes = []

    # loop over all datasets
    for i, ds in enumerate(args.datasets):
        ds_path = os.path.join(args.data_dir, ds)
        ds_path_out = ds_path + SUFFIX_FILTERED
        load_scores = load_scores_fns[ds]
        print_stats = print_stats_fns[ds]
        filter_relevant = filter_relevant_fns[ds]
        print_frm("Processing dataset %d/%d: %s" % (i+1, len(args.datasets), ds_path))

        # loading scores
        print_frm("Loading scores")
        df_scores, scores, slicenumbers = load_scores(os.path.join(ds_path, SCORES_FILE))

        # print stats before filtering
        print_frm("Score stats before filtering")
        print_stats(df_scores.reset_index(), scores)

        # load images and filter missing data
        print_frm("Loading images and filtering missing data")
        (t1_volumes, t2_volumes), (df_scores, scores, slicenumbers) = filter_relevant(ds_path, df_scores, scores,
                                                                                      slicenumbers)

        # setup domains
        domains = [ds for i in range(len(t1_volumes))]

        # print stats after filtering
        print_frm("Score stats after filtering")
        print_stats(df_scores.reset_index(), scores)

        if args.merge:

            # keep track of data
            all_scores.append(scores)
            all_slicenumbers.append(slicenumbers)
            all_domains.append(domains)
            all_t1_volumes.append(t1_volumes)
            all_t2_volumes.append(t2_volumes)

        else:

            # save images
            print_frm("Saving images")
            mkdir(ds_path_out)
            save(t1_volumes, os.path.join(ds_path_out, T1_PP_FILE))
            save(t2_volumes, os.path.join(ds_path_out, T2_PP_FILE))

            # save scores
            print_frm("Saving scores")
            save(scores, os.path.join(ds_path_out, SCORES_PP_FILE))
            save(slicenumbers, os.path.join(ds_path_out, SLICENUMBERS_PP_FILE))
            save(domains, os.path.join(ds_path_out, DOMAINS_PP_FILE))

    if args.merge:

        # merge all data together
        print_frm("Merging")
        merged_scores = all_scores[0]
        merged_slicenumbers = all_slicenumbers[0]
        merged_domains = all_domains[0]
        merged_t1_volumes = all_t1_volumes[0]
        merged_t2_volumes = all_t2_volumes[0]
        for k in range(1, len(all_scores)):
            merged_scores = tuple([merged_scores[i]+all_scores[k][i] for i in range(len(merged_scores))])
            merged_slicenumbers = tuple([merged_slicenumbers[i]+all_slicenumbers[k][i] for i in range(len(merged_slicenumbers))])
            merged_domains = merged_domains + all_domains[k]
            merged_t1_volumes = merged_t1_volumes + all_t1_volumes[k]
            merged_t2_volumes = merged_t2_volumes + all_t2_volumes[k]

        # save images
        print_frm("Saving images")
        ds_path_out = os.path.join(args.data_dir, MERGED_DIR)
        mkdir(ds_path_out)
        save(merged_t1_volumes, os.path.join(ds_path_out, T1_PP_FILE))
        save(merged_t2_volumes, os.path.join(ds_path_out, T2_PP_FILE))

        # save scores
        print_frm("Saving scores and metadata")
        save(merged_scores, os.path.join(ds_path_out, SCORES_PP_FILE))
        save(merged_slicenumbers, os.path.join(ds_path_out, SLICENUMBERS_PP_FILE))
        save(merged_domains, os.path.join(ds_path_out, DOMAINS_PP_FILE))

    print_frm('Done!')
