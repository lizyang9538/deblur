import torch
import operations_ltd as op
import torch.nn as nn
import torch.nn.functional as F
from math import ceil, log2
import config_ltd as config
import pdb
import numpy as np


class CNN(nn.Module):
    def __init__(self,
                in_ch,
                out_ch,
                num_layers=1,
                kernel_size=3,
                bias=True):
        super(CNN, self).__init__()

        def conv2d(in_size, out_size, kernel_size, bias):
            return nn.Conv2d(in_size, out_size, kernel_size, bias, padding=1,
                            padding_mode='replicate')
        conv_layers = [conv2d(in_ch, out_ch, kernel_size, bias)]
        for layer in range(num_layers - 1):
            conv_layers.append(conv2d(out_ch, out_ch, kernel_size, bias))
        self.layers = nn.ModuleList(conv_layers)

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x


class NonlinearizationUnit(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(NonlinearizationUnit, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.tanh = nn.Tanh()
        self.cnn = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, input):
        x = input
        x = self.tanh(x)
        x = self.cnn(x)
        return x


class NonlinearizationNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(NonlinearizationNet, self).__init__()
        self.nonlinear_layer = NonlinearizationUnit(in_ch, out_ch)

    def forward(self, input):
        x = input
        x = self.nonlinear_layer(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch, feature_extractor_stages=2):
        super(FeatureExtractor, self).__init__()
        self.cnn = CNN(in_ch, config.feature_extractor_hidden_dim)
        nonlinear_layers = [
        NonlinearizationNet(config.feature_extractor_hidden_dim,
                            config.feature_extractor_hidden_dim)
                            for _ in range(feature_extractor_stages-1)]

        self.layers = nn.ModuleList(nonlinear_layers)

        self.linear_0 = nn.Conv2d(config.feature_extractor_hidden_dim
                                    , out_ch
                                    , kernel_size=1
                                    , bias=False)
        self.linear_1 = nn.Conv2d(config.feature_extractor_hidden_dim
                                    , out_ch
                                    , kernel_size=1
                                    , bias=False)
        self.tanh = nn.Tanh()

    def forward(self, input):
        x = input
        x = self.cnn(x)
        for layer in self.layers:
            x = layer(x)
        x = self.tanh(x)
        x_head = self.linear_0(x)
        y_head = self.linear_1(x)
        return x_head, y_head

def estimator(input_x, input_y, beta):
    # TODO: add beta
    beta_tensor = torch.from_numpy(np.array([beta]))

    H_x, W_x = input_x.shape[-2], input_x.shape[-1]
    H_y, W_y = input_y.shape[-2], input_y.shape[-1]
    fft_size = [2**(int(log2(ceil(H_x + H_y - 1)))),2**(int(log2(ceil(W_x + W_y - 1))))]
    Fx, Fy = op.fft2(input_x, fft_size), op.fft2(input_y, fft_size)
    numerator = torch.sum(op.conj_mul(Fx, Fy), dim=-4, keepdim=True)
    denominator = torch.sum(op.csquare(Fx), dim=-3, keepdim=True)
    sol = op.ifft2(torch.div(numerator, denominator.unsqueeze(dim=-1)), fft_size)
    return sol

class KernelEstimator(nn.Module):
    def __init__(self, beta_k):
        super(KernelEstimator, self).__init__()
        self.beta_k = beta_k

    def forward(self, x_head, y_head, mode='full'):
        return estimator(x_head, y_head, self.beta_k)

class ImageEstimator(nn.Module):
    def __init__(self, beta_x):
        super(ImageEstimator, self).__init__()
        self.beta_x = beta_x

    def forward(self, k_head, y):
        return estimator(k_head, y, self.beta_x)

class DeblurUnit(nn.Module):
    def __init__(self):
        super(DeblurUnit, self).__init__()
        self.feature_extractor = FeatureExtractor(config.im_channels,
                    config.out_channels,
                    feature_extractor_stages=config.feature_extractor_stages)
        self.kernel_estimator = KernelEstimator(config.beta_k)
        self.image_estimator = ImageEstimator(config.beta_x)

    def forward(self, img):
        x_head, y_head = self.feature_extractor(img)
        k_head = self.kernel_estimator(x_head, y_head)
        return self.image_estimator(k_head, img)

class LearningToDeblur(nn.Module):
    def __init__(self, num_stacks):
        super(LearningToDeblur, self).__init__()
        self.networks = nn.ModuleList([DeblurUnit() for _ in range(num_stacks)])

    def forward(self, blur_in):
        x = blur_in
        for net in self.networks:
            x = net(x)
        return x
