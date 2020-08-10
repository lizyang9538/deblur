import cv2
import numpy as np
from glob import glob
from os.path import join
from torch.utils.data import Dataset


class BlurImgDataset(Dataset):
    def __init__(self, root_dir):
        self.blur_files = sorted(glob(join(root_dir, 'blurs', '*.png')))
        self.ker_files = sorted(glob(join(root_dir, 'kernels', '*.png')))
        self.img_files = sorted(glob(join(root_dir, 'images', '*.png')))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        ker = cv2.imread(self.ker_files[idx])
        if ker.ndim == 3:
            ker = cv2.cvtColor(ker, cv2.COLOR_BGR2GRAY)
        ker = np.expand_dims(ker.astype('float32'), axis=0)
        ker /= ker.sum()
        blur = cv2.imread(self.blur_files[idx])
        if blur.ndim == 3:
            blur = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        Hk, Wk = ker.shape[-2], ker.shape[-1]
        pad_size = ((Hk // 2, Hk - Hk // 2 - 1), (Wk // 2, Wk - Wk // 2 - 1))
        blur = np.pad(blur, pad_size, mode='linear_ramp')
        blur = np.expand_dims(blur.astype('float32'), axis=0) / 255.
        img = cv2.imread(self.img_files[idx])
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img.astype('float32'), axis=0) / 255.
        return blur, ker, img
