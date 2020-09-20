import cv2
import torch
import operations_ltd as op
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve
from model_ltd import KernelEstimator as EstimateKernel


def conv2(image, kernel):
    '''
    2D image convolution in 'valid' scheme
    '''
    hk, wk = kernel.shape
    h0, w0 = hk // 2, wk // 2
    kernel /= kernel.sum()
    if image.ndim == 3:  # color image
        kernel = np.expand_dims(kernel, axis=-1)
    blur = convolve(image, kernel, mode='constant')
    hb, wb = image.shape[0] - hk + 1, image.shape[1] - wk + 1
    blur = blur[h0: h0 + hb, w0: w0 + wb]
    blur[blur < 0.] = 0.
    blur[blur > 1.] = 1.
    return blur


if __name__ == '__main__':
    blur = cv2.imread('../train/blur_09930.png')
    if blur.ndim == 3:
        blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    blur = np.expand_dims(blur.astype('float32'), axis=0) / 255.
    blur = torch.from_numpy(blur).unsqueeze(0)
    img = cv2.imread('../train/image_09930.png')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img.astype('float32'), axis=0) / 255.
    img = torch.from_numpy(img).unsqueeze(0)
    gx, gy = op.image_gradient(img)
    ker = cv2.imread('../train/kernel_09930.png')
    if ker.ndim == 3:
        ker = cv2.cvtColor(ker, cv2.COLOR_BGR2GRAY)
    ker = np.expand_dims(ker.astype('float32'), axis=0)
    ker = torch.from_numpy(ker / ker.sum())
    estimate_kernel = EstimateKernel()
    s = 0.5
    gx_s = op.rescale(gx, s)
    gy_s = op.rescale(gy, s)
    blur_s = op.rescale(blur, s)
    ker_est = estimate_kernel(blur_s, gx_s, gy_s)
    ker_s = op.resize(torch.unsqueeze(ker, 1), ker_est.shape[2:])
    bx, by = op.image_gradient(blur)
    gx_est = op.deconv(bx, ker)
    plt.imshow(gx_est[0, 0], cmap='gray')
    # plt.subplot(121)
    # plt.imshow(ker_est[0, 0], cmap='gray')
    # plt.subplot(122)
    # plt.imshow(ker_s[0, 0], cmap='gray')
    plt.show()
