import sys, os

import h5py
import torch
import torch.utils
import torch.utils.data
import numpy as np
import nibabel as nib

from constants import *


def load_data(split, path, size, ind):
    field = np.empty(size, dtype=np.float32)
    suscep = np.empty(size, dtype=np.float32)
    mask = np.empty(size, dtype=np.float32)

    with h5py.File(path, 'r') as f:
        f['input'].read_direct(field, np.s_[ind])
        f['label'].read_direct(suscep, np.s_[ind])
        f['mask_t'].read_direct(mask, np.s_[ind])
        field = torch.from_numpy(field).permute(3, 0, 1, 2)
        suscep = torch.from_numpy(suscep).permute(3, 0, 1, 2)
        mask = torch.from_numpy(mask).permute(3, 0, 1, 2)

        if split == 'val':
            roi_mask = np.empty(size, dtype=np.float32)
            f['roi_mask'].read_direct(roi_mask, np.s_[ind])
            roi_mask = torch.from_numpy(roi_mask).permute(3, 0, 1, 2)
            return field, suscep, mask, roi_mask
        else:
            return field, suscep, mask


def load_data_res(split, path, size, ind):
    field = np.empty(size, dtype=np.float32)
    suscep = np.empty(size, dtype=np.float32)
    conven = np.empty(size, dtype=np.float32)
    mask = np.empty(size, dtype=np.float32)

    with h5py.File(path, 'r') as f:
        f['input'].read_direct(field, np.s_[ind])
        f['label'].read_direct(suscep, np.s_[ind])
        f['mask_t'].read_direct(mask, np.s_[ind])
        field = torch.from_numpy(field).permute(3, 0, 1, 2)
        suscep = torch.from_numpy(suscep).permute(3, 0, 1, 2)
        mask = torch.from_numpy(mask).permute(3, 0, 1, 2)

        if split == 'val':
            roi_mask = np.empty(size, dtype=np.float32)
            # f['roi_mask'].read_direct(roi_mask, np.s_[ind])
            f['conven'].read_direct(conven, np.s_[ind])
            roi_mask = torch.from_numpy(roi_mask).permute(3, 0, 1, 2)
            # return field, suscep, mask, roi_mask, conven
            return field, suscep, mask, conven
        else:
            return field, suscep, mask


def load_dipole():
    print('Begin Loading dipole kernel')

    with h5py.File(dipole_path, 'r') as f:
        dipole_kernel = np.asarray(f["Dpatch_train"])[:batch_size, :, :, :, :]

    dipole_kernel = np.transpose(dipole_kernel, (0, 4, 1, 2, 3))
    complex_dipole = torch.from_numpy(np.stack([dipole_kernel, np.zeros_like(dipole_kernel)], axis=-1))

    print('Dipole kernel loaded')
    return complex_dipole


class QSMDataset(torch.utils.data.Dataset):
    def __init__(self, split, get_statistics=True):
        if split[:3] == 'res':
            res, split = split.split('_')
            assert split in ['train', 'val', 'test']
            self.split = split
            self.data_type = 'res'
            self.data_path = eval('res_' + self.split + '_data_path')
            self.data_size = (64, 64, 64, 1) if self.split == 'train' else (176, 176, 160, 1)
            with h5py.File(self.data_path, 'r') as f:
                self.length = f['input'].len()
            if get_statistics:
                if split == 'val':
                    with h5py.File(res_train_data_path, 'r') as f:
                        self.X_mean = f['input_mean_total'].__array__().item()
                        self.X_std = f['input_std_total'].__array__().item()
                        self.RES_mean = f['label_mean_total'].__array__().item()    # label means residual
                        self.RES_std = f['label_std_total'].__array__().item()
                    with h5py.File(res_val_data_path, 'r') as f:
                        self.Y_mean = f['label_mean_total'].__array__().item()  # label means real suscep
                        self.Y_std = f['label_std_total'].__array__().item()

                else:
                    with h5py.File(res_train_data_path, 'r') as f:
                        self.X_mean = f['input_mean_total'].__array__().item()
                        self.X_std = f['input_std_total'].__array__().item()
                        self.Y_mean = f['label_mean_total'].__array__().item()      # label means residual
                        self.Y_std = f['label_std_total'].__array__().item()

        else:
            assert split in ['train', 'val', 'test']
            self.split = split
            self.data_type = 'normal'
            self.data_path = eval(self.split + '_data_path')
            self.data_size = (64, 64, 64, 1) if self.split == 'train' else (176, 176, 160, 1)
            with h5py.File(self.data_path, 'r') as f:
                self.length = f['input'].len()
            if get_statistics:
                with h5py.File(train_data_path, 'r') as f:
                    self.X_mean = f['input_mean_total'].__array__().item()
                    self.X_std = f['input_std_total'].__array__().item()
                    self.Y_mean = f['label_mean_total'].__array__().item()
                    self.Y_std = f['label_std_total'].__array__().item()

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        if self.data_type == 'normal':
            if self.split == 'train' or self.split == 'test':
                field, suscep, mask = load_data(self.split, self.data_path, self.data_size, ind)
                field = (field - self.X_mean) / self.X_std
                suscep = (suscep - self.Y_mean) / self.Y_std
                return field, suscep, mask
            else:   # validation
                field, suscep, mask, roi_mask = load_data(self.split, self.data_path, self.data_size, ind)
                field = (field - self.X_mean) / self.X_std
                suscep = (suscep - self.Y_mean) / self.Y_std
                return field, suscep, mask, roi_mask

        elif self.data_type == 'res':
            if self.split == 'train' or self.split == 'test':
                field, suscep, mask = load_data_res(self.split, self.data_path, self.data_size, ind)
                field = (field - self.X_mean) / self.X_std
                suscep = (suscep - self.Y_mean) / self.Y_std
                return field, suscep, mask
            else:
                field, suscep, mask, conven = load_data_res(self.split, self.data_path, self.data_size, ind)
                field = (field - self.X_mean) / self.X_std
                suscep = (suscep - self.Y_mean) / self.Y_std
                return field, suscep, mask, conven


def save_nii(data, save_folder, name, mask=None):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    nifti_affine = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]], dtype=np.float)

    data = data.cpu().detach().squeeze().numpy()
    data = np.fliplr(data)
    data = np.pad(data, ((2, 2), (6, 7), (6, 7)), mode='constant')
    if mask is not None:
        data = data * mask
    nifti = nib.Nifti1Image(data, affine=nifti_affine)
    nib.save(nifti, os.path.join(save_folder, name + '.nii'))
