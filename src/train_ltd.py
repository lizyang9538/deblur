#!/usr/bin/env python3
import torch
import math
import config_ltd as config
import os
import operations_ltd as op
import matplotlib.pyplot as plt
from shutil import copyfile
from model_ltd import LearningToDeblur as Deblur
from data import BlurImgDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

if __name__ == '__main__':
    chkpt_dir = os.path.join(config.checkpoint_dir, 'fine_tune') \
        if config.fine_tune else \
        os.path.join(config.checkpoint_dir, 'stage_%02d' % config.stage)
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    model = Deblur(config.num_stacks)
    model = torch.nn.DataParallel(model)

    init_lr = config.fine_tune_lr if config.fine_tune else config.init_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.step_size)

    trainset = BlurImgDataset(config.train_dir)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.batch_size)
    writer = SummaryWriter(os.path.join(config.log_dir, 'fine_tune')
                           if config.fine_tune else os.path.join(
                               config.log_dir, 'stage_%02d' % config.stage))

    it = 0
    for epoch in range(config.max_epoch):
        for i, (blur, ker, img) in enumerate(trainloader):
            # blur = blur.to(config.device)
            im_pred = model(blur)
            # ker_gt = ker.to(config.device)
            optimizer.zero_grad()
            loss = op.shift_invariant_mse(im_pred, img)
            loss.backward()
            if math.isnan(loss):
                __import__('ipdb').set_trace()

            optimizer.step()
            writer.add_scalar('loss', loss.detach(), it)
            if it % config.write_every == 0:
                writer.add_image('im_pred', make_grid(
                    im_pred, normalize=True), it)
                writer.add_image('im_gt', make_grid(
                    img, normalize=True), it)
                # writer.add_image('ker_pred', make_grid(
                #     k_pred, normalize=True), it)
                # writer.add_image('ker_gt', make_grid(
                #     ker_gt, normalize=True), it)
            print('Iteration %d, loss %.4f' % (it, loss))
            it += 1
        if config.fine_tune:
            model_path = os.path.join(chkpt_dir, 'model_%03d.pth' % epoch)
            torch.save(model.state_dict(), model_path)
            copyfile(model_path, os.path.join(chkpt_dir, 'model_best.pth'))
        else:
            model_path = os.path.join(chkpt_dir, 'cnn_%03d.pth' % epoch)
            torch.save(model.state_dict(), model_path)
            copyfile(model_path, os.path.join(chkpt_dir, 'cnn_latest.pth'))
        print('Model saved to {}'.format(model_path))
        scheduler.step()
