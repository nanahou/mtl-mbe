#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 29/7/19 3:09 PM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : gen_noisy8k.py


"""
Create 8k then resample to 16k for training super-resolution model.
"""

import argparse
import librosa
from scipy.signal import resample
import soundfile as sf

# ----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--in-dir', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy16_clean16/tr/clean/',
                    help='folder where input files are located')
parser.add_argument('--out', default= '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tr/clean_re16k/',
                    help='folder where output files are located')
parser.add_argument('--scale', type=int, default=2,
                    help='scaling factor')
parser.add_argument('--sr', type=int, default=16000, help='audio sampling rate')

parser.add_argument('--tr_list_path', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/VB_28spk_tr_list.txt')
parser.add_argument('--val_list_path', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/VB_28spk_val_list.txt')
parser.add_argument('--tt_list_path', default='/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/tt_list.txt')

args = parser.parse_args()

# ----------------------------------------------------------------------------

def resamp_data(args):
    # Make a list of all files to be processed
    with open(args.tr_list_path) as f:
        file_list = f.readlines()

    num_files = len(file_list)

    for j, item in enumerate(file_list):
        if j % 10 == 0:
            print('%d/%d' % (j, num_files))

        item = item.strip()
        # load audio file
        x, fs = librosa.load(args.in_dir + item, sr=args.sr)
        assert fs == args.sr

        # generate low-res version
        x_lr = resample(x, len(x)//args.scale)
        x_hr = resample(x_lr, len(x))
        assert len(x_hr) == len(x)

        f_name = args.out + item

        sf.write(f_name, x_hr, fs)


if __name__ == '__main__':
    # create train
    resamp_data(args)

