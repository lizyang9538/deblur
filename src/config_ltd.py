#!/usr/bin/env python3
import torch

im_channels = 1  # 3 for rgb, 1 for grayscale
max_epoch = 40
init_lr = 1e-3
fine_tune_lr = 1e-5
device = torch.device('cuda')
step_size = 20
batch_size = 4
write_every = 1000
max_shift = 10
num_layers = 6
stage = 1
scales = [0.5, 1., 1., 1.]
fine_tune = False
train_dir = '../data/train'
# train_dir = 'train'
feature_extractor_hidden_dim = 8
im_channels = 1
out_channels = 2
feature_extractor_hidden_dim = 8
feature_extractor_stages = 2
beta_k = 1e-4
beta_x = 1e-4
num_stacks = 2
# checkpoint_dir = '/content/drive/My Drive/Deblur/models_ltd'
checkpoint_dir = '../models'
log_dir = 'logs'
