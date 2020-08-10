import torch
import config_ltd as config
import torch.nn.functional as F
from math import ceil, log2


def real(c):
    '''
    Extract real part of complex tensor c
    '''
    return c[..., 0]


def real_mul(r, c):
    '''
    Multiply real tensor r with complex tensor c
    '''
    return r.unsqueeze(dim=-1)*c


def mul(c1, c2):
    '''
    Complex multiplication between c1 and c2
    '''
    r1, i1 = c1[..., 0], c1[..., 1]
    r2, i2 = c2[..., 0], c2[..., 1]
    r = r1*r2 - i1*i2
    c = r1*i2 + i1*r2
    return torch.stack([r, c], dim=-1)


def conj(c):
    '''
    Complex conjugation of complex tensor c
    '''
    return torch.stack([c[..., 0], -c[..., 1]], dim=-1)


def conj_mul(c1, c2):
    '''
    Complex conjugate of c1 and multiplication with c2
    '''
    r1, i1 = c1[..., 0], -c1[..., 1]
    r2, i2 = c2[..., 0], c2[..., 1]
    r = r1*r2 - i1*i2
    c = r1*i2 + i1*r2
    return torch.stack([r, c], dim=-1)


def csquare(c):
    '''
    Square of absolute values of complex numbers
    '''
    return c[..., 0]**2 + c[..., 1]**2


def pad_to(original, size):
    '''
    Post-pad last two dimensions to "size"
    '''
    original_size = original.size()
    pad = [0, size[1] - original_size[-1],
           0, size[0] - original_size[-2]]
    return F.pad(original, pad)


def fft2(signal, size=None):
    '''
    Fast Fourier transform on the last two dimensions
    '''
    padded = signal if size is None else pad_to(signal, size)
    return torch.rfft(padded, signal_ndim=2)


def ifft2(signal, size=None):
    '''
    Inverse fast Fourier transform on the last two dimensions
    '''
    return torch.irfft(signal, signal_ndim=2, signal_sizes=size)


def conv2fft(image, kernel, mode='full'):
    '''
    Fast valid convolution using fast Fourier transform
    '''
    Hi, Wi = image.shape[-2], image.shape[-1]
    Hk, Wk = kernel.shape[-2], kernel.shape[-1]
    fft_size = [int(2**ceil(log2(Hi + Hk - 1))),
                int(2**ceil(log2(Wi + Wk - 1)))]
    Fx = fft2(image, fft_size)
    Fk = fft2(kernel.unsqueeze(1), fft_size)
    xk = ifft2(mul(Fk, Fx), fft_size)
    if mode == 'valid':
        return xk[..., Hk - 1: Hi, Wk - 1: Wi]
    elif mode == 'full':
        return xk[..., : Hi + Hk - 1, : Wi + Wk - 1]


def conv2(image, kernel, mode='same'):
    '''
    2D sample-wise discrete convolution on every image channels
    '''
    _, Hk, Wk = kernel.shape
    if mode == 'same':
        input = F.pad(image, [Wk // 2, Wk - Wk // 2 - 1,
                              Hk // 2, Hk - Hk // 2 - 1],
                      mode='replicate')
    elif mode == 'valid':
        input = image
    else:
        raise ValueError('Either same or valid convolutions are supported!')
    output = []
    for channel, weight in zip(input, kernel):
        channel = channel.unsqueeze(0)
        weight = weight.repeat(len(channel), 1, 1).unsqueeze(0)
        weight = torch.flip(weight, [-2, -1])
        conv_out = F.conv2d(channel, weight)
        output.append(conv_out[0])
    return torch.stack(output)


def image_shift(im, shift):
    '''
    Shift on the last two dimensions
    '''
    sr, sc = shift[0].item(), shift[1].item()
    dim = im.ndimension()
    if dim == 2:
        im = im.unsqueeze(0).unsqueeze(0)
    elif dim == 3:
        im = im.unsqueeze(0)
    if sr > 0:
        im = F.pad(im[..., :-sr, :], (0, 0, sr, 0), mode='replicate')
    else:
        im = F.pad(im[..., -sr:, :], (0, 0, 0, -sr), mode='replicate')
    if sc > 0:
        im = F.pad(im[..., :-sc], (sc, 0, 0, 0), mode='replicate')
    else:
        im = F.pad(im[..., -sc:], (0, -sc, 0, 0), mode='replicate')
    if dim == 2:
        im = im[0, 0]
    elif dim == 3:
        im = im[0]
    return im


def estimate_kernel(blur, gx, gy, eps=1.):
    fft_size = (int(2**ceil(log2(blur.shape[-2]))),
                int(2**ceil(log2(blur.shape[-1]))))
    bx, by = image_gradient(blur)
    Fbx, Fby = fft2(bx, size=fft_size), fft2(by, size=fft_size)
    Fgx, Fgy = fft2(gx, size=fft_size), fft2(gy, size=fft_size)
    num = conj_mul(Fgx, Fbx) + conj_mul(Fgy, Fby)
    den = csquare(Fgx) + csquare(Fgy) + eps
    k = ifft2(num / den.unsqueeze(dim=-1), size=fft_size)
    Hk = blur.shape[-2] - gx.shape[-2] + 1
    Wk = blur.shape[-1] - gx.shape[-1] + 1
    k = k[..., :Hk, :Wk]
    return k


def image_gradient(img):
    dx = torch.tensor([[-1, 1]], dtype=img.dtype,
                      device=img.device).repeat(len(img), 1, 1)
    dy = torch.tensor([[1], [-1]], dtype=img.dtype,
                      device=img.device).repeat(len(img), 1, 1)
    gx = conv2(img, dx, 'same')
    gy = conv2(img, dy, 'same')
    return gx, gy


def deconv(blur, kernel, rho=0.01):
    Hk, Wk = kernel.shape[-2], kernel.shape[-1]
    Hi = blur.shape[-2] - Hk + 1
    Wi = blur.shape[-1] - Wk + 1
    # pad_size = [Wk // 2, Wk - Wk // 2 - 1, Hk // 2, Wk - Hk // 2 - 1]
    # blur = F.pad(blur, pad_size, mode='constant')
    fft_size = (int(2**ceil(log2(blur.shape[-2]))),
                int(2**ceil(log2(blur.shape[-1]))))
    dx = torch.tensor([[1., -1.], [0., 0.]], device=config.device)
    dy = torch.tensor([[1., 0.], [-1., 0.]], device=config.device)
    Fdx = fft2(dx, size=fft_size)
    Fdy = fft2(dy, size=fft_size)
    Fy = fft2(blur, size=fft_size)
    Fk = fft2(kernel, size=fft_size)
    den = csquare(Fk) + rho * (csquare(Fdx) + csquare(Fdy))
    x = ifft2(conj_mul(Fk, Fy) / den.unsqueeze(dim=-1), fft_size)
    x = x[..., :Hi, :Wi]
    return x


def get_min_shift(ref, tar, max_shift):
    dist = ifft2(conj_mul(fft2(tar), fft2(ref)))
    shifts = torch.arange(-max_shift, max_shift + 1)
    ri, ci = shifts % dist.shape[0], shifts % dist.shape[1]
    roi = dist[ri][:, ci]
    u = roi.argmax().item() // (2*max_shift + 1) - max_shift
    v = roi.argmax().item() % (2*max_shift + 1) - max_shift
    return torch.IntTensor((u, v))


def resize(img, size):
    return F.interpolate(img, size)


def rescale(img, ratio=1.):
    Hi, Wi = int(ratio * img.shape[-2]), int(ratio * img.shape[-1])
    img = resize(img, [Hi, Wi])
    return img


def shift_invariant_mse(pred, img):
    img = resize(img, pred.shape[-2:])
    ims = torch.empty_like(img)
    for n in range(len(img)):
        # Shift compensation
        min_shift = get_min_shift(
            pred[n, 0].detach(), img[n, 0], config.max_shift)
        ims[n] = image_shift(img[n], -min_shift)
    mse = torch.nn.MSELoss(reduction='sum')
    return mse(pred, ims) / len(img)
