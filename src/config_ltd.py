#!/usr/bin/env python3
import torch

use_colab = False
im_channels = 1  # 3 for rgb, 1 for grayscale
max_epoch = 45
init_lr = 1e-3
fine_tune_lr = 1e-5
device = torch.device('cuda:0')
step_size = 20
batch_size = 4
write_every = 1000
max_shift = 10
num_layers = 6
stage = 1
scales = [0.5, 1., 1., 1.]
fine_tune = False
train_dir = '../data/train'
test_dir = '../data/test'
train_dir_colab = 'train'
test_dir_colab = 'test'
feature_extractor_hidden_dim = 8
im_channels = 1
out_channels = 2
feature_extractor_hidden_dim = 8
feature_extractor_stages = 2
beta_k = 1e-4
beta_x = 1e-4
num_stacks = 2
checkpoint_dir = '../models'
checkpoint_dir_colab = '/content/drive/My Drive/Deblur/models_ltd'
log_dir = 'logs'
