#!/usr/bin/env python3
import torch
import os
import config_ltd as config
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from data import BlurImgDataset
from model_ltd import LearningToDeblur as Deblur
import operations_ltd as op


def imshow(image_list, title):
    if not image_list:
        return
    image_tensor = torch.stack(image_list)
    N, C, H, W = image_tensor.shape
    grid = make_grid(image_tensor.reshape(N * C, 1, H, W), normalize=True)
    grid_img = grid.permute(1, 2, 0)
    plt.figure(title)
    plt.imshow(np.squeeze(grid_img.cpu().detach().numpy()))
    plt.show()


if __name__ == '__main__':
    testset = BlurImgDataset(config.train_dir_colab if config.use_colab else config.train_dir)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=1)

    with torch.no_grad():
        model = Deblur(config.num_stacks)
        # checkpoint = torch.load('checkpoints/fine_tune/model_best.pth')
        for s, net in enumerate(model.networks):
            if config.use_colab:
                net.to(config.device)
            checkpoint = torch.load(os.path.join(
                config.checkpoint_dir_colab, 'stage_%02d/cnn_latest.pth' % (s + 1)) \
                if config.use_colab else os.path.join(
                    config.checkpoint_dir, 'stage_%02d/cnn_latest.pth' % (s + 1)))
            net.load_state_dict(checkpoint)
        model = torch.nn.DataParallel(model)
        if config.use_colab:
            model.to(config.device)

        ker_preds, ker_gts, img_gts = [], [], []
        for i, (blur, ker, img) in enumerate(testloader):
            if config.use_colab:
                blur = blur.to(config.device)
                ker = ker.to(config.device)
            k_pred = model(blur, ker.shape[-2:])
            loss = op.shift_invariant_mse(k_pred, ker)
            ker_preds.append(k_pred[0])
            ker_gts.append(ker[0])
            img_gts.append(img[0])
            print('Finished processing sample %d, loss %.2f' % (i, loss))
            if i > 64:
                break
        nrow = int(round(np.sqrt(len(testloader))))
        imshow(img_gts, title='Groundtruth images')
        imshow(ker_preds, title='Estimated kernels')
        imshow(ker_gts, title='Groundtruth kernels')
