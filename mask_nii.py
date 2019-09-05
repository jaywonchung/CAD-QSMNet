import os

import h5py
import nibabel as nib
import numpy as np


# Get inferences
checkpoint_path = input('Input checkpoint directory path: ')
checkpoint_folders = [os.path.join(checkpoint_path, f) for f in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, f)) and os.path.isdir(os.path.join(checkpoint_path, f, 'Inference'))]
checkpoint_folders.sort()
print('\n'.join(checkpoint_folders))

if not os.path.exists(os.path.join(os.path.dirname(checkpoint_path), 'masked_sim')):
    os.mkdir(os.path.join(os.path.dirname(checkpoint_path), 'masked_sim'))

# Get challenge data mask
finaldataset = os.path.join(input('Input path to FinalDataset folder (exclude FinalDataset): '), 'FinalDataset')
mask_path = os.path.join(finaldataset, 'Sim1', 'MaskBrainExtracted.nii.gz')
mask1 = np.asanyarray(nib.nifti1.load(mask_path).dataobj)
mask_path = os.path.join(finaldataset, 'Sim2', 'MaskBrainExtracted.nii.gz')
mask2 = np.asanyarray(nib.nifti1.load(mask_path).dataobj)

# Read inferences and mask them
nifti_affine = np.array([[1,0,0,1], [0,1,0,1], [0,0,1,1], [0,0,0,1]], dtype=np.float)
for folder in checkpoint_folders:
    for sim_number in [1, 2]:
        sim = np.asanyarray(nib.nifti1.load(os.path.join(folder, 'Inference', f'Sim{sim_number}_inference.nii.gz')).dataobj)
        sim = sim * eval(f'mask{sim_number}')
        nifti = nib.Nifti1Image(sim, affine=nifti_affine)
        if not os.path.exists(os.path.join(os.path.dirname(checkpoint_path), 'masked_sim', os.path.basename(folder))):
            os.mkdir(os.path.join(os.path.dirname(checkpoint_path), 'masked_sim', os.path.basename(folder)))
        nib.save(nifti, os.path.join(os.path.dirname(checkpoint_path), 'masked_sim', os.path.basename(folder), f'_Sim{sim_number}_Step1.nii'))