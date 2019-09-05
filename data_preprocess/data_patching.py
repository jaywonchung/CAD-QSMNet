import scipy.io
import numpy as np
import h5py
import time

start_time = time.time()

'''
File Path
'''

FILE_PATH_DATA = 'training_set_before_patch.mat'
FILE_PATH_MASK = 'mask_total.mat'

'''
Constant Variables
'''
matrix_cut = [172, 176, 159]
patch_counts = [7, 8, 6]  # Order of Dimensions: [x, y, z]
ps = 64

aug_count = 10       # max value = 10
dataset_count = 5   # max value = 7
mask_count = 5   # max value = 5

'''
Code Start
'''

strides = [(matrix_cut[i] - ps) // (patch_counts[i] - 1) for i in range(3)]

mat = scipy.io.loadmat(FILE_PATH_DATA)
mask_h5 = h5py.File(FILE_PATH_MASK, 'r')

mask = {}
for i in range(5):
    mask['mask' + str(i + 1)] = np.array(mask_h5['mask' + str(i + 1)])
del mask_h5

# Create Result File
result_file = h5py.File('result.hdf5', 'w')

# Patch the input file ----------------------------------------------------------------

print("patching start")

patches = []
for dataset_num in range(1, dataset_count + 1):
    for idx in range(aug_count):
        print("doing")
        for i in range(patch_counts[0]):
            for j in range(patch_counts[1]):
                for k in range(patch_counts[2]):
                    patches.append(mat['multiphs' + str(dataset_num)][
                                   i * strides[0]:i * strides[0] + ps,
                                   j * strides[1]:j * strides[1] + ps,
                                   k * strides[2]:k * strides[2] + ps,
                                   idx])

patches = np.stack(patches)

result_file.create_dataset('temp_i', data=patches)

del patches

# Patch the label file --------------------------------------------------------------------

patches = []
for dataset_num in range(1, dataset_count + 1):
    for idx in range(aug_count):
        print("doing2")
        for i in range(patch_counts[0]):
            for j in range(patch_counts[1]):
                for k in range(patch_counts[2]):
                    patches.append(mat['multicos' + str(dataset_num)][
                                   i * strides[0]:i * strides[0] + ps,
                                   j * strides[1]:j * strides[1] + ps,
                                   k * strides[2]:k * strides[2] + ps,
                                   idx])

patches = np.stack(patches)

result_file.create_dataset('temp_l', data=patches)


del patches


# Patch the mask file --------------------------------------------------------------
patches = []

for mask_num in range(1, mask_count + 1):
    for idx in range(aug_count):
        for i in range(patch_counts[0]):
            for j in range(patch_counts[1]):
                for k in range(patch_counts[2]):
                    patches.append(mask['mask' + str(mask_num)][
                                   i * strides[0]:i * strides[0] + ps,
                                   j * strides[1]:j * strides[1] + ps,
                                   k * strides[2]:k * strides[2] + ps,
                                   idx])


# patches = np.stack(patches)

# result_file.create_dataset('mask_i', data=patches)

# del patches
del mat
result_file.close()

print("Total Time: {:.3f}".format(time.time()-start_time))