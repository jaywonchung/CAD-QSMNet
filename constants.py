import torch

data_folder = '../Data/'

C = {
    'data': {
        'data_folder': data_folder,
        'train_data_path': data_folder + 'Train_0704_Contrast.mat',
        'val_data_path': data_folder + 'Validation_0710.mat',
        'test_data_path': data_folder + 'Test_0718.mat',
        'test_ROI_path': data_folder + 'Test_ROI_mask.mat',
        'inference_data_folder': '/data/list3/Personal_folder/woojin/QSM_challenge/DatasetsStep2',
        'inference_data_path': data_folder + 'Challenge_stage2_sim{}snr{}_{}.mat',
        'inference_ROI_path': data_folder + 'Challenge_stage1_ROI_mask.mat',
        'dipole_path': data_folder + 'dipole_t_64.mat',
        'res_train_data_path': data_folder + 'Train_0719_iLSQR_11760.mat',
        'res_val_data_path': data_folder + 'Validation_Residual_iLQSR_0720.mat',
        'res_inference_data_path': data_folder + 'Challenge_stage2_sim{}snr{}_res.mat',
    },
    'model': {
        'identifier': 'dual-frame-unet'
    },
    'loss': {
        'one': {
            'w1': 1,
            'w2': 0.01,
            'w3': 0
        },
        'two': {
            'w1': 0,
            'w2': 0.01,
            'w3': 1
        },
        'one_to_two': 200 # Switch gears at 'one_to_two'th epoch (epoch number counted from 1)
    },
    'train': {
        'batch_size': 12,
        'train_epochs': 200,
        'learning_rate': 5e-3,
        'print_every': 1,
        'save_every': 5,
        'save_base_path': 'Checkpoints/',
        'device_name': 'cuda:0'
    }
}

data_folder = C['data']['data_folder']
train_data_path = C['data']['train_data_path']
val_data_path = C['data']['val_data_path']
test_data_path = C['data']['test_data_path']
test_ROI_path = C['data']['test_ROI_path']
inference_data_folder = C['data']['inference_data_folder']
inference_data_path = C['data']['inference_data_path']
inference_ROI_path = C['data']['inference_ROI_path']
dipole_path = C['data']['dipole_path']
res_train_data_path = C['data']['res_train_data_path']
res_val_data_path = C['data']['res_val_data_path']
res_inference_data_path = C['data']['res_inference_data_path']

identifier = C['model']['identifier']

one_w1 = C['loss']['one']['w1']
one_w2 = C['loss']['one']['w2']
one_w3 = C['loss']['one']['w3']

two_w1 = C['loss']['two']['w1']
two_w2 = C['loss']['two']['w2']
two_w3 = C['loss']['two']['w3']

one_to_two = C['loss']['one_to_two']

batch_size = C['train']['batch_size']
train_epochs = C['train']['train_epochs']
learning_rate = C['train']['learning_rate']
print_every = C['train']['print_every']
save_every = C['train']['save_every']
save_base_path = C['train']['save_base_path']
device_name = C['train']['device_name']
device = torch.device(C['train']['device_name']) if torch.cuda.is_available() else torch.device('cpu')
