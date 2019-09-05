import os
import sys
import json
import warnings

import h5py
from scipy.io import savemat
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import nibabel as nib

from loss import *
from constants import *
from data_utils import QSMDataset, save_nii


def inf(checkpoint_file, file=None, numpy=None, tensor=None, device=None):
    # Check arguments
    if not (file or numpy or tensor):
        raise ValueError('Specify at least one type of input.')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f'Cannot find checkpoint file :{checkpoint_file}.')
    config_path = os.path.join(os.path.dirname(checkpoint_file), 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Cannot find config.json in {os.path.dirname(checkpoint_file)}.')
    if device is None:
        from constants import device
    
    # Fetch input data statistics
    with open(config_path, 'r') as f:
        config = json.load(f)
        train_data = config['data']['res_train_data_path']
    with h5py.File(train_data, 'r') as f:
        b_mean = f['input_mean_total'].__array__().item()
        b_std = f['input_std_total'].__array__().item()
        y_mean = f['label_mean_total'].__array__().item()
        y_std = f['label_std_total'].__array__().item()
    
    # Load model from checkpoint
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'source code')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
    network = checkpoint['model'].to(device)
    del checkpoint

    # Perform inference
    ret = []
    for input_type, input_data in enumerate([file, numpy, tensor]):
        # Get data
        if input_type == 0 and input_data: # input_data is a file path. we should first fetch from it.
            assert isinstance(input_data, str)
            extension = input_data.split('.')[-1]
            if extension == 'mat':
                with h5py.File(input_data, 'r') as f:
                    key = f.keys().__iter__().__next__()
                    input_data = torch.from_numpy(f[key].__array__())
            elif extension == 'gz' or extension == 'nii':
                nifti = nib.nifti1.load(input_data)
                input_data = torch.from_numpy(np.asanyarray(nifti.dataobj))
                del nifti
            else:
                raise NotImplementedError(f'Extension {extension} not supported.')
        elif input_type == 1 and numpy is not None:   # input_data is np.array
            assert isinstance(input_data, np.ndarray)
            input_data = torch.from_numpy(input_data)
            assert isinstance(input_data, torch.Tensor)
        elif input_type == 2 and tensor is not None:  # input_data is torch.Tensor
            pass
        else:
            continue

        # Inference
        with torch.no_grad():
            if len(input_data.shape) < 5:
                input_data = input_data.squeeze().unsqueeze(0).unsqueeze(0).float()
            input_data = (input_data.to(device) - b_mean) / b_std
            chi = network(input_data) * y_std + y_mean
            chi = chi.cpu()
        if input_type == 0 or input_type == 1:
            chi = chi.numpy()
        ret.append(chi.squeeze())
    
    return tuple(ret) if len(ret) > 1 else ret[0]


def inference(model_architecture, experience_name, model_name):
    with torch.no_grad():
        # Prepare save path
        base_path = os.path.join('Checkpoints', model_architecture, experience_name)
        inference_save_path = os.path.join(base_path, f'{experience_name}-{model_name}', 'Inference')
        evaluation_save_path = os.path.join(base_path, f'{experience_name}-{model_name}', 'Evaluation')
        config_path = os.path.join(base_path, 'config.json')
        save_mat_dict = dict()
        if not os.path.exists(os.path.join(base_path, f'{experience_name}-{model_name}')):
            os.mkdir(os.path.join(base_path, f'{experience_name}-{model_name}'))
        if not os.path.exists(inference_save_path):
            os.mkdir(inference_save_path)
        if not os.path.exists(evaluation_save_path):
            os.mkdir(evaluation_save_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Cannot find {config_path}.')

        # Load model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'source code')
            checkpoint = torch.load(os.path.join(base_path, model_name), map_location='cpu')
        network = checkpoint['model']
        del checkpoint
        network = network.to(device).eval()

        # Fetch input data statistics
        with open(config_path, 'r') as f:
            config = json.load(f)
            train_data = config['data']['res_train_data_path']
        with h5py.File(train_data, 'r') as f:
            b_mean = f['input_mean_total'].__array__().item()
            b_std = f['input_std_total'].__array__().item()
            y_mean = f['label_mean_total'].__array__().item()
            y_std = f['label_std_total'].__array__().item()

        # For plotting
        methods = [('iLSQR', 'orange'), ('MEDI', 'green'), ('QSMNet', 'blue'), ('Ours', 'black')]
        ROI_name = ['Deep Grey Matter']*5 + ['White Matter']*5 + ['Center Region']*2 + ['Cortical Grey Matter']*5 + ['Calcification']*3 + ['Vein']*5

        # Challenge Data
        for sim_number in [1, 2]:
            print(f'Sim{sim_number} image:\nDoing inference...', end='')

            # Load and format challenge data
            with h5py.File(inference_data_path.format(sim_number), 'r') as f:
                b = f['temp'].__array__()
            
            b = (torch.from_numpy(b) - b_mean) / b_std
            b = b.unsqueeze(0).unsqueeze(0).float().to(device)

            # Inference and de-normalization
            chi_res = network(b)
            chi_res = chi_res * y_std + y_mean
            with h5py.File(os.path.join(data_folder, 'Conventional', f'Sim{sim_number}_iLSQR_fix.mat')) as f:
                recon = f[f'Sus_ilsqr'].__array__()
                recon = torch.from_numpy(recon)
            chi = chi_res.cpu() + recon 

            print('done\nComputing statistics and plotting...', end='')

            # Overall mean and std plot
            plt.clf()
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.xaxis.set_visible(False)
            ax.set_title(f'Sim{sim_number} | Overall Brain | {model_name}')
            ax.set_ylabel('Susceptibility Statistics')

            for num, (method, color) in enumerate(methods):
                if method == 'Ours':
                    sim_mean = torch.mean(chi).item()
                    sim_std = torch.std(chi).item()
                else:
                    with h5py.File(os.path.join(data_folder, 'Conventional', f'Sim{sim_number}_{method}_fix.mat')) as f:
                        recon = f[f'Sus_{method.lower()}'].__array__()
                    recon = torch.from_numpy(recon)
                    sim_mean = torch.mean(recon).item()
                    sim_std = torch.std(recon).item()
                ax.errorbar(num+1, sim_mean, sim_std, capsize=10, markersize=7, marker='o', color=color, ecolor=color, mfc=color, mec=color, label=method)

            plt.legend(bbox_to_anchor=(0., -0.12, 1., -0.202), loc=8, ncol=5, mode="expand", borderaxespad=0., fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig(os.path.join(inference_save_path, f'Sim{sim_number}_overall.png'))

            # Mean and std plot of each ROI
            plt.clf()
            fig, ax = plt.subplots(1, len(ROI_name), figsize=(90, 4))
            for name, axis in zip(ROI_name, ax):
                axis.xaxis.set_visible(False)
                axis.set_title(name)
            plt.suptitle(f'Sim{sim_number} | 25 ROIs | {model_name}')
            ax[0].set_ylabel('Susceptibility Statistics')

            for mask_num in range(25):
                rm = np.empty((160, 192, 192, 1), dtype=np.float32)
                with h5py.File(inference_ROI_path, 'r') as f:
                    f['roi_mask'].read_direct(rm, np.s_[mask_num])
                rm = torch.from_numpy(rm).squeeze().byte()
                for num, (method, color) in enumerate(methods):
                    if method == 'Ours':
                        recon_mean = torch.mean(chi.squeeze()[rm]).item()
                        recon_std = torch.std(chi.squeeze()[rm]).item()
                    else:
                        with h5py.File(os.path.join(data_folder, 'Conventional', f'Sim{sim_number}_{method}_fix.mat')) as f:
                            recon = f[f'Sus_{method.lower()}'].__array__()
                        recon = torch.from_numpy(recon)
                        recon_mean = torch.mean(recon[rm]).item()
                        recon_std = torch.std(recon[rm]).item()
                    ax[mask_num].errorbar(num+1, recon_mean, recon_std, capsize=10, markersize=7, marker='o', color=color, ecolor=color, mfc=color, mec=color, label=method)

            plt.tight_layout(rect=(0, 0, 1, 0.92))
            plt.savefig(os.path.join(inference_save_path, f'Sim{sim_number}_ROI.png'))

            print('done\nSaving as NIFTI...', end='')

            # Save MATLAB array
            save_mat_dict[f'chi{sim_number}'] = chi.numpy().squeeze()

            # Save NIFTI image
            save_nii(chi, inference_save_path, f'Sim{sim_number}_inference')

            print('done')
        
        print("Saving MATLAB file...", end='')
        savemat(os.path.join(inference_save_path, 'Sim12_inference.mat'), mdict=save_mat_dict)
        print('done')

#         # Test Set Data
#         save_mat_dict = dict()
#         test_loader = DataLoader(QSMDataset('test', get_statistics=False), batch_size=1, shuffle=False)
#         test_loader.__dict__['dataset'].X_mean = b_mean
#         test_loader.__dict__['dataset'].X_std = b_std
#         test_loader.__dict__['dataset'].Y_mean = y_mean
#         test_loader.__dict__['dataset'].Y_std = y_std

#         methods = [('iLSQR', 'orange'), ('StarQSM', 'cyan'), ('QSMNet', 'blue'), ('Ours', 'black')]
#         plot_names = ['Mean/Std', 'RMSE', 'ddRMSE']
#         image_names = ['Real', 'Phantom', 'Phantom+Calcification', 'Phantom+Linearity']
#         test_roi_names = ['Deep Grey Matter']*5 + ['Uniform White Matter']*3 + ['Veins + Neighboor']*5 + ['Only Calcification', 'Only Calcification Neighboor', 'Calcification + Neighboor']

#         y = None
#         chi = None
#         for batch, (b, y, _) in enumerate(test_loader):
#             print(f'Test{batch//4+9:02d} {image_names[batch%4]} image:\nDoing inference...', end='')

#             b = b.to(device)
#             chi = network(b)

#             y = y * y_std + y_mean
#             chi = chi * y_std + y_mean
#             chi = chi.cpu()

#             print('done\nComputing statistics and plotting...', end='')

#             # Overall mean, std, RMSE, ddRMSE
#             plt.clf()
#             plt.close('all')
#             fig = plt.figure(figsize=(15, 4))
#             ax = [None for _ in range(3)]
#             ax[0] = fig.add_subplot(1, 3, 1)
#             ax[1] = fig.add_subplot(1, 3, 2)
#             ax[2] = fig.add_subplot(1, 3, 3, sharey=ax[1])

#             for plot_name, axis in zip(plot_names, ax):
#                 axis.figsize = (5, 4)
#                 axis.xaxis.set_visible(False)
#                 axis.set_title(plot_name)
#             plt.suptitle(f'Test{batch//4+9} | {image_names[batch%4]} | Overall Brain | {model_name}')

#             for method_num, (method, color) in enumerate(methods):
#                 if method == 'Ours':
#                     sim_mean = torch.mean(chi).item()
#                     sim_std = torch.std(chi).item()
#                     sim_rmse = loss_rmse(chi, y).item()
#                     sim_ddrmse = loss_ddrmse(chi, y).item()
#                 else:
#                     recon = np.empty((176, 176, 160, 1), dtype=np.float32)
#                     with h5py.File(os.path.join(data_folder, 'Conventional', f'Test_{method}.mat')) as f:
#                         f[method].read_direct(recon, np.s_[batch])
#                     recon = torch.from_numpy(recon).permute(3, 0, 1, 2).unsqueeze(0)
#                     sim_mean = torch.mean(recon).item()
#                     sim_std = torch.std(recon).item()
#                     sim_rmse = loss_rmse(recon, y).item()
#                     sim_ddrmse = loss_ddrmse(recon, y).item()
#                 ax[0].errorbar(method_num+1, sim_mean, sim_std, capsize=10, markersize=7, marker='o', color=color, ecolor=color, mfc=color, mec=color, label=method)
#                 ax[1].plot(method_num+1, sim_rmse, marker='o', markersize=7, color=color)
#                 ax[2].plot(method_num+1, sim_ddrmse, marker='o', markersize=7, color=color)
            
#             label_mean = torch.mean(y).item()
#             label_std = torch.std(y).item()
#             ax[0].errorbar(len(methods)+1, label_mean, label_std, capsize=10, markersize=7, marker='o', color='brown', ecolor='brown', mfc='brown', mec='brown', label='GT')

#             ax[0].legend(bbox_to_anchor=(0., -0.12, 1., -0.202), loc=8, ncol=5, mode="expand", borderaxespad=0., fancybox=True, shadow=True)
#             plt.tight_layout(rect=(0, 0, 1, 0.92))
#             plt.savefig(os.path.join(evaluation_save_path, f'Test{batch//4+9:02d}_{image_names[batch%4]}_overall.png'))

#             # ROI mean, std, RMSE, ddRMSE
#             plt.clf()
#             plt.close('all')
#             fig = plt.figure(figsize=(80, 13))
#             ax = [[None for _ in range(len(test_roi_names))] for _ in range(len(plot_names))]
#             for row, plot_name in enumerate(plot_names):
#                 for col, roi_name in enumerate(test_roi_names):
#                     ax[row][col] = fig.add_subplot(len(plot_names), len(test_roi_names), row * len(test_roi_names) + col + 1)
#                     ax[row][col].xaxis.set_visible(False)
#                     ax[row][col].set_title(roi_name)
#                     ax[row][col].figsize = (5, 4)
#                 ax[row][0].set_ylabel(plot_name)
#             plt.suptitle(f'Test{batch//4+9} | {image_names[batch%4]} | 16 ROIs | {model_name}')

#             for roi_num, roi_name in enumerate(test_roi_names):
#                 rm = np.empty((176, 176, 160, 1))
#                 with h5py.File(test_ROI_path, 'r') as f:
#                     f['roi_mask'].read_direct(rm, np.s_[(batch//4)*16+roi_num])
#                 rm = torch.from_numpy(rm).squeeze().unsqueeze(0).unsqueeze(0).byte()
#                 for method_num, (method, color) in enumerate(methods):
#                     if method == 'Ours':
#                         recon_mean = torch.mean(chi[rm]).item()
#                         recon_std = torch.std(chi[rm]).item()
#                         recon_rmse = loss_rmse(chi[rm], y[rm]).item()
#                         recon_ddrmse = loss_ddrmse(chi[rm], y[rm]).item()
#                     else:
#                         recon = np.empty((176, 176, 160, 1), dtype=np.float32)
#                         with h5py.File(os.path.join(data_folder, 'Conventional', f'Test_{method}.mat')) as f:
#                             f[method].read_direct(recon, np.s_[batch])
#                         recon = torch.from_numpy(recon).permute(3, 0, 1, 2).unsqueeze(0)
#                         recon_mean = torch.mean(recon[rm]).item()
#                         recon_std = torch.std(recon[rm]).item()
#                         recon_rmse = loss_rmse(recon[rm], y[rm]).item()
#                         recon_ddrmse = loss_ddrmse(recon[rm], y[rm]).item()
#                     ax[0][roi_num].errorbar(method_num+1, recon_mean, recon_std, capsize=10, markersize=7, marker='o', color=color, ecolor=color, mfc=color, mec=color, label=method)
#                     ax[1][roi_num].plot(method_num+1, recon_rmse, marker='o', markersize=7, color=color)
#                     ax[2][roi_num].plot(method_num+1, recon_ddrmse, marker='o', markersize=7, color=color)
#                 label_mean = torch.mean(y[rm]).item()
#                 label_std = torch.std(y[rm]).item()
#                 ax[0][roi_num].errorbar(len(methods)+1, label_mean, label_std, capsize=10, markersize=7, marker='o', color='brown', ecolor='brown', mfc='brown', mec='brown', label='GT')

#             plt.tight_layout(rect=(0, 0, 1, 0.92))            
#             plt.savefig(os.path.join(evaluation_save_path, f'Test{batch//4+9:02d}_{image_names[batch%4]}_ROI.png'))

#             print('done\nSaving as NIFTI...', end='')

#             # Save MATLAB array
#             save_mat_dict[f'y{batch//4+9}{image_names[batch%4]}'] = y.numpy().squeeze()
#             save_mat_dict[f'chi{batch//4+9}{image_names[batch%4]}'] = chi.numpy().squeeze()

#             # Save NIFTI image
#             save_nii(y.squeeze(), evaluation_save_path, f'Test_label{batch//4+9:02d}_{image_names[batch%4]}')
#             save_nii(chi.squeeze(), evaluation_save_path, f'Test_inference{batch//4+9:02d}_{image_names[batch%4]}')

#             print('done')
        
#         print("Saving MATLAB file...", end='')
#         savemat(os.path.join(evaluation_save_path, 'Test_inference_label.mat'), mdict=save_mat_dict)
#         print('done')
        print('All done!')


if __name__=='__main__':
    if len(sys.argv) != 2:
        raise ValueError("Provide the checkpoint file path via commandline argument.")
    
    if os.path.exists(sys.argv[-1]):
        print(f'Checkpoint verified! Begin inference on {device}')
        _, model_architecture, experience_name, checkpoint_name = sys.argv[-1].split('/')
        inference(model_architecture, experience_name, checkpoint_name)
    else:
        raise ValueError(f"No such file: {sys.argv[-1]}.")