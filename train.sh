#!/usr/bin/env bash

set -eu

#cpt_dir=exp/TasBWETasNet
cpt_dir=exp_test2/
epochs=100
# constrainted by GPU number & memory
batch_size=2
cache_size=2

[ $# -ne 2 ] && echo "Script error: $0 <gpuid> <cpt-id>" && exit 1

./nnet/train.py \
  --gpu $1 \
  --epochs $epochs \
  --batch-size $batch_size \
  --cache-size $cache_size \
  --checkpoint $cpt_dir/$2 \
  > $2.train.log 2>&1
