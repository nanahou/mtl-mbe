fs = 16000
chunk_len = 2  # (s)
chunk_size = chunk_len * fs
num_spks = 1

# network configure
nnet_conf = {
    "L": 16,
    "N": 512,
    "X": 8,
    "R": 3,
    "B": 128,
    "H": 512,
    "P": 3,
    "norm": "gLN",
    "num_spks": num_spks,
    "non_linear": "relu"
}

# data configure:
train_dir = "/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/tr/"
dev_dir = "/home/hounana/pytorch/enhancement/conv-tasnet-noisyBWE-multitask/data/cv/"

train_data = {
    "mix_scp": train_dir + "mix.scp",
    "ref1_scp": [train_dir + "ref1.scp"],
    "ref2_scp": [train_dir + "ref2.scp"],
    "sample_rate":
    fs,
}

dev_data = {
    "mix_scp": dev_dir + "mix.scp",
    "ref1_scp": [dev_dir + "ref1.scp"],
    "ref2_scp": [dev_dir + "ref2.scp"],
    "sample_rate": fs,
}

# trainer config
adam_kwargs = {
    "lr": 1e-4,
    "weight_decay": 1e-5,
}

trainer_conf = {
    "optimizer": "adam",
    "optimizer_kwargs": adam_kwargs,
    "min_lr": 1e-8,
    "patience": 2,
    "factor": 0.5,
    "logging_period": 200  # batch number
}