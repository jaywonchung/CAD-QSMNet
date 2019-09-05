import math
import torch
import torch.nn.functional as F
from constants import *

'''
Partial Functions to Calculate the Loss Terms
'''


def _l1error(x1, x2):
    return torch.mean(torch.abs(x1 - x2))


def _l2error(x1, x2):
    return torch.mean((x1 - x2) ** 2)


def _l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def _chi_to_b(chi, b, d, m, b_mean, b_std, y_mean, y_std):
    # Restore from normalization
    chi = chi * y_std + y_mean
    b = b * b_std + b_mean

    # Multiply dipole kernel in Fourier domain
    chi_fourier = torch.rfft(chi, signal_ndim=3, onesided=False)
    b_hat_fourier = chi_fourier * d
    b_hat = torch.irfft(b_hat_fourier, signal_ndim=3, onesided=False)

    # Multiply masks
    b = b * m
    b_hat = b_hat * m

    return b, b_hat


def _sobel_kernel():
    s = [
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ],
        [
            [-1, -2, -1],
            [-2, -4, -2],
            [-1, -2, -1]
        ]
    ]
    s = torch.FloatTensor(s)
    sx = s
    sy = s.permute(1, 2, 0)
    sz = s.permute(2, 0, 1)
    ss = torch.stack([sx, sy, sz]).unsqueeze(1)

    return ss


def _gaussian_window():
    window = torch.empty(5, 5, 5)
    for x in range(5):
        for y in range(5):
            for z in range(5):
                radiusSquared = (x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2
                window[x, y, z] = math.exp(-radiusSquared / (2 * 1.5 ** 2))
    return window / torch.sum(window)


'''
Loss Terms for Training

Descriptions of all parameters are written in the last function: total_loss
'''


def loss_l1(chi, y):
    return _l1error(chi, y)


def loss_l2(chi, y):
    return _l2error(chi, y)


def loss_model(b, b_hat):
    return _l1error(b, b_hat)


def loss_gradient(b, b_hat, sobel):
    difference = F.conv3d(b - b_hat, sobel, padding=1)
    return torch.mean(torch.abs(difference))


def loss_rmse_QSM(chi, y):
    return 100. * _l2norm(chi - y) / _l2norm(y)


def loss_rmse(chi, y):
    return torch.sqrt(F.mse_loss(chi, y))


def loss_ddrmse(chi, y):
    # Linear Regression for each pair in patch: y = a + b * chi
    if len(chi.shape) == 1:
        chi_mean = torch.mean(chi)
        y_mean = torch.mean(y)
        b = torch.sum((chi - chi_mean) * (y - y_mean)) / torch.sum((chi - chi_mean) ** 2)
        a = torch.mean(y) - b * torch.mean(chi)
    else:
        kwargs = {'dim': [1, 2, 3, 4], 'keepdim': True}
        chi_mean = torch.mean(chi, **kwargs)
        y_mean = torch.mean(y, **kwargs)
        b = torch.sum((chi - chi_mean) * (y - y_mean), **kwargs) / torch.sum((chi - chi_mean) ** 2, **kwargs)
        a = torch.mean(y, **kwargs) - b * torch.mean(chi, **kwargs)

    return torch.sqrt(torch.mean((b * chi - y + a) ** 2 * 2 / (b ** 2 + 1)))


def loss_ssim(img1, img2):
    # Get gaussian window
    if not hasattr(loss_ssim, 'window'):
        loss_ssim.window = _gaussian_window().unsqueeze(0).unsqueeze(0).to(device)

    # Scale image pixel range to [0, 255]
    min_img = min(torch.min(img1).item(), torch.min(img2).item())

    img1[img1 != 0] = img1[img1 != 0] - min_img
    img2[img2 != 0] = img2[img2 != 0] - min_img

    max_img = max(torch.max(img1).item(), torch.max(img2).item())

    img1 = 255 * img1 / max_img
    img2 = 255 * img2 / max_img

    # Calculate SSIM components
    mu1 = F.conv3d(img1, loss_ssim.window, padding=2)
    mu2 = F.conv3d(img2, loss_ssim.window, padding=2)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, loss_ssim.window, padding=2) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, loss_ssim.window, padding=2) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, loss_ssim.window, padding=2) - mu1_mu2

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return torch.mean(ssim_map[img1 != 0])


def loss_hfen(img1, img2):
    if not hasattr(loss_hfen, 'filter'):
        filter_size = 15
        sigma = 1.5
        size = 7

        x = torch.arange(-size, size + 1).reshape(filter_size, 1, 1)
        x = x.repeat(1, filter_size, filter_size).double()
        y = torch.arange(-size, size + 1).reshape(1, filter_size, 1)
        y = y.repeat(filter_size, 1, filter_size).double()
        z = torch.arange(-size, size + 1).reshape(1, 1, filter_size)
        z = z.repeat(filter_size, filter_size, 1).double()

        h = torch.exp(-(x ** 2 / (2 * sigma ** 2) + y ** 2 / (2 * sigma ** 2) + z ** 2 / (2 * sigma ** 2)))
        h = h / torch.sum(h)

        arg = x ** 2 / sigma ** 4 + y ** 2 / sigma ** 4 + z ** 2 / sigma ** 4 - (
                    1 / sigma ** 2 + 1 / sigma ** 2 + 1 / sigma ** 2)
        H = arg * h
        H = H - torch.sum(H) / (2 * 1.5 ** 2 + 1)
        loss_hfen.filter = H.float().unsqueeze(0).unsqueeze(0).to(device)

    img1_log = F.conv3d(img1, loss_hfen.filter, padding=7)
    img2_log = F.conv3d(img2, loss_hfen.filter, padding=7)

    return loss_rmse_QSM(img1_log, img2_log)


def loss_psnr(chi, y):
    return 10 * torch.log10((torch.max(y) - torch.min(y)) ** 2 / F.mse_loss(chi, y))


def total_loss(chi, y, b, d, m, b_mean, b_std, y_mean, y_std):
    # Sobel kernel for 3D image gradient loss
    # if not hasattr(total_loss, 'sobel'):
    #     total_loss.sobel = _sobel_kernel().to(device)

    # B field for model loss and 3D image gradient loss
    # b, b_hat = _chi_to_b(chi, b, d, m, b_mean, b_std, y_mean, y_std)

    loss1 = loss_l1(chi, y)
    # loss2 = loss_hfen(chi * m, y * m)
    # loss3 = loss_l2(chi, y)

    #w1 = eval(total_loss.phase + '_w1')
    #w2 = eval(total_loss.phase + '_w2')
    #w3 = eval(total_loss.phase + '_w3')

    return loss1, {'L1 Loss': loss1.detach().item()}
