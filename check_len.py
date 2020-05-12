#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 30/7/19 3:53 PM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : check_len.py


import argparse
import librosa
from scipy.signal import resample
import soundfile as sf
import numpy as np

# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--path_d', default='/data/disk3/hounana/Valentini-Botinhao_16k/clean_trainset_28spk_wav/',
                    help='folder where input files are located')
parser.add_argument('--path_c', default= '/data/disk3/hounana/Valentini-Botinhao_16k/clean_trainset_28spk_wav/',
                    help='folder where output files are located')
parser.add_argument('--scale', type=int, default=2,
                    help='scaling factor')
parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')

parser.add_argument('--tr_list_path', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/VB_28spk_tr_list.txt')
parser.add_argument('--val_list_path', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/VB_28spk_val_list.txt')
parser.add_argument('--tt_list_path', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/tt_list.txt')

args = parser.parse_args()

# ----------------------------------------------------------------------------

def check_len(args):
    # Make a list of all files to be processed
    with open(args.tr_list_path) as f:
        file_list = f.readlines()

    num_files = len(file_list)

    for j, item in enumerate(file_list):
        # if j % 10 == 0:
        #     print('%d/%d' % (j, num_files))

        item = item.strip()
        # load audio file
        x_d, fs_d = librosa.load(args.path_d + item, sr=args.sr)
        assert fs_d == args.sr
        x_c, fs_c = librosa.load(args.path_c + item, sr=args.sr)
        assert fs_c == args.sr

        if len(x_d) != len(x_c):
            print('%s/%d/%d' % (item, len(x_d), len(x_c)))
            # min_len = np.min(len(x_d), len(x_c))
            # x_d = x_d[:min_len]
            # x_c = x_c[:min_len]
            # assert len(x_d) == len(x_c)
            #
            # f_name_d = args.path_d + item
            # sf.write(f_name_d, x_d, fs_d)
            #
            # f_name_c = args.path_c + item
            # sf.write(f_name_c, x_c, fs_c)


if __name__ == '__main__':
    # create train
    check_len(args)
