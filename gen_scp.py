#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 16/7/19 11:23 AM
# @Author  : HOU NANA
# @Site    : http://github.com/nanahou
# @File    : gen_scp.py
'''
 mix is the input (noisy re16k)
 ref1 is the target (clean re16k)
 ref2 is the target (clean true 16k)
'''

tr_path_d = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tr/noisy/'
tr_path_c1 = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tr/clean_re16k/'
tr_path_c2 = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tr/clean/'

cv_path_d = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/cv/noisy/'
cv_path_c1 = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/cv/clean_re16k/'
cv_path_c2 = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/cv/clean/'

tt_path_d = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tt/noisy/'
tt_path_c1 = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tt/clean_re16k/'
tt_path_c2 = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/noisy_re16k_clean16k/tt/clean/'

tr_list_path = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/VB_28spk_tr_list.txt'
val_list_path = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/VB_28spk_val_list.txt'
tt_list_path = '/data/disk3/hounana/Valentini-Botinhao_16k/formatted_28spk/tt_list.txt'

tr_mix_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tr/mix.scp'
tr_ref1_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tr/ref1.scp'
tr_ref2_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tr/ref2.scp'

val_mix_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/cv/mix.scp'
val_ref1_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/cv/ref1.scp'
val_ref2_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/cv/ref2.scp'

tt_mix_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tt/mix.scp'
tt_ref1_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tt/ref1.scp'
tt_ref2_scp = '/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tt/ref2.scp'


tr_list= list()
with open(tr_list_path) as ftr:
    for line in ftr:
      filename = line.strip()
      tr_list.append(filename)


with open(tr_mix_scp, 'w') as f:
    for item in tr_list:
        f.write("%s %s\n" % (item, tr_path_d+item))

with open(tr_ref1_scp, 'w') as f:
    for item in tr_list:
        f.write("%s %s\n" % (item, tr_path_c1+item))

with open(tr_ref2_scp, 'w') as f:
    for item in tr_list:
        f.write("%s %s\n" % (item, tr_path_c2+item))
