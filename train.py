#!/usr/bin/env python

# wujian@2018

import os
import pprint
import argparse
import random

from libs.trainer import SiSnrTrainer
from libs.dataset import make_dataloader
from libs.utils import dump_json, get_logger

# from conv_tas_net import ConvTasNet
# from Tas_BWE_net import TasBweNet
from Tas_BWE_Tas_net import TasBweTasNet
from conf import trainer_conf, nnet_conf, train_data, dev_data, chunk_size

logger = get_logger(__name__)


def run(args):
    gpuids = tuple(map(int, args.gpus.split(",")))

    nnet = TasBweTasNet(**nnet_conf)
    trainer = SiSnrTrainer(
        nnet,
        gpuid=gpuids,
        checkpoint=args.checkpoint,
        resume=args.resume,
        **trainer_conf)

    data_conf = {
        "train": train_data,
        "dev": dev_data,
        "chunk_size": chunk_size
    }
    for conf, fname in zip([nnet_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, args.checkpoint, fname)

    train_loader = make_dataloader(
        train=True,
        data_kwargs=train_data,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        chunk_size=chunk_size,
        use_mulaw=args.mulaw)
    dev_loader = make_dataloader(
        train=False,
        data_kwargs=dev_data,
        batch_size=args.batch_size,
        cache_size=args.cache_size,
        chunk_size=chunk_size,
        use_mulaw=args.mulaw)

    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start ConvTasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--gpus",
        type=str,
        default="3",
        help="Training on which GPUs(one or more, egs: 0, \"0,1\")")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/exp_tmp/',
        # required=True,
        help="Directory to dump models")
    parser.add_argument(
        "--resume",
        type=str,
        default=False,
        help="Exist model to resume training from")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of utterances in each batch")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=1,
        help="Number of chunks cached in dataloader")
    parser.add_argument(
        "--mulaw",
        type=str,
        default=True,
        help="whether use mulaw to preprocess data")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))

    run(args)
