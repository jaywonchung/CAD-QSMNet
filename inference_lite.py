import os, io, sys
import warnings
import json

import h5py
from scipy.io import savemat
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import nibabel as nib
import matlab.engine

from loss import *
from constants import *
from data_utils import QSMDataset, save_nii


def inference(model_architecture, experience_name, model_name, method, called_in_train=False, network=None, writer=None, step=None):
    with torch.no_grad():
        # Configuration
        if called_in_train:
            dataset_folder = inference_data_folder
        else:
            dataset_folder = input('Input path to stage2 data folder (press ENTER to use default path /data/list3/Personal_folder/woojin/QSM_challenge/DatasetsStep2): ') or '/data/list3/Personal_folder/woojin/QSM_challenge/DatasetsStep2'

        # Prepare save path
        base_path = os.path.join('Checkpoints', model_architecture, experience_name)
        inference_save_path = os.path.join(base_path, f'{experience_name}-{model_name}-{method}')
        config_path = os.path.join(base_path, 'config.json')

        if not os.path.exists(inference_save_path):
            os.mkdir(inference_save_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'Cannot find {config_path}.')
        
        # Initialize dictionary object to save challenge metrics
        metrics = dict()
        
        if not called_in_train:
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
            train_data = config['data']['train_data_path']
        with h5py.File(train_data, 'r') as f:
            b_mean = f['input_mean_total'].__array__().item()
            b_std = f['input_std_total'].__array__().item()
            y_mean = f['label_mean_total'].__array__().item()
            y_std = f['label_std_total'].__array__().item()
        
        # Prepare MATLAB engine
        matlab_engine = matlab.engine.start_matlab()
        matlab_engine.addpath(matlab_engine.genpath(os.path.join(dataset_folder, 'functions4challenge')))

        # Perform inference, calculate metrics
        for sim_number in [1, 2]:
            for snr_number in [1, 2]:
                if not called_in_train:
                    print(f'Sim{sim_number}Snr{snr_number} image:')

                # Load and format challenge data
                try:
                    with h5py.File(inference_data_path.format(sim_number, snr_number, method), 'r') as f:
                        b = f['temp'].__array__()
                    if not called_in_train:
                        print('\tDoing inference...', end='')
                        sys.stdout.flush()
                except:
                    if not called_in_train:
                        print('\tChallenge input file not found.')
                    continue

                b = (torch.from_numpy(b) - b_mean) / b_std
                b = b.unsqueeze(0).unsqueeze(0).float().to(device)

                # Inference and de-normalization
                chi = network(b)
                b = b * b_std + b_mean
                chi = chi * y_std + y_mean
                chi = chi.cpu()

                # Load mask
                mask_path = os.path.join(dataset_folder, f'Sim{sim_number}Snr{snr_number}', 'MaskBrainExtracted.nii.gz')
                mask = np.asanyarray(nib.nifti1.load(mask_path).dataobj)

                # Save NIFTI image (masked)
                save_nii(chi, inference_save_path, f'_Sim{sim_number}Snr{snr_number}_Step2', mask)

                if called_in_train:
                    y = np.asanyarray(nib.nifti1.load(os.path.join(dataset_folder, f'Sim{sim_number}Snr{snr_number}', 'GT', 'Chi.nii.gz')).dataobj)
                    y = y[2:-2, 6:-7, 6:-7]
                    y = torch.from_numpy(np.fliplr(y).copy())

                    input_images = (torch.clamp(b[0, 0, :, :, ::20].permute(2, 0, 1).unsqueeze(1), min=-0.05, max=0.05) * 10) + 0.5
                    output_images = (torch.clamp(chi[0, 0, :, :, ::20].permute(2, 0, 1).unsqueeze(1), min=-0.1, max=0.1) * 5) + 0.5
                    label_images = (torch.clamp(y[:, :, ::20].permute(2, 0, 1).unsqueeze(1), min=-0.1, max=0.1) * 5) + 0.5
                    writer.add_image(f'Challenge sim{sim_number}snr{snr_number} {method}/Input', make_grid(input_images, padding=20), step)
                    writer.add_image(f'Challenge sim{sim_number}snr{snr_number} {method}/Output', make_grid(output_images, padding=20), step)
                    writer.add_image(f'Challenge sim{sim_number}snr{snr_number} {method}/Label', make_grid(label_images, padding=20), step)

                # Calculate metrics with MATLAB and save to dictionary object
                if not called_in_train:
                    print('done\n\tCalculating metrics...', end='')
                    sys.stdout.flush()
                metrics[f'Sim{sim_number}Snr{snr_number}'] = matlab_engine.challenge_inference(inference_save_path, dataset_folder, sim_number, snr_number, stdout=io.StringIO())

                if called_in_train:
                    for simsnr, dictionary in metrics.items():
                        for key, value in dictionary.items():
                            writer.add_scalar(f'Challenge {simsnr} {method}/{key}', value, step)

                if not called_in_train:
                    print('done')
                    sys.stdout.flush()

        # Save challenge metric dictionary to json
        with open(os.path.join(inference_save_path, f'challenge_metrics.json'), 'w') as fp:
            json.dump(metrics, fp)

        # Display final metrics
        if not called_in_train: 
            print('Challenge metrics:')
            print(json.dumps(metrics, indent=2))

            print('All done')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise ValueError("Provide the checkpoint file path via commandline argument.")

    if os.path.exists(sys.argv[-1]):
        print(f'Checkpoint verified! Begin inference on {device}')
        _, model_architecture, experience_name, checkpoint_name = sys.argv[-1].split('/')
        method = ['VSHARP', 'PDF'][int(input('Input 0 for VSHARP / 1 for PDF: '))]
        inference(model_architecture, experience_name, checkpoint_name, method)
    else:
        raise ValueError(f"No such file: {sys.argv[-1]}.")
