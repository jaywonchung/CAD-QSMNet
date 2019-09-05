import sys, os
import json
from datetime import datetime

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ModuleNotFoundError:
    from tensorboardX import SummaryWriter

from constants import *
from model import QSMNet
from loss import *
from inference_lite import inference
from data_utils import load_dipole, QSMDataset, save_nii


def evaluate_model(name, network, loader, step, save_path, writer=None):
    with torch.no_grad():
        print(f'Evaluating model with {name} dataset...')
        dataset_name = name[0].upper() + name[1:]

        b_mean = loader.__dict__['dataset'].X_mean
        b_std = loader.__dict__['dataset'].X_std
        y_mean = loader.__dict__['dataset'].Y_mean
        y_std = loader.__dict__['dataset'].Y_std

        network.eval()
        for batch, (b, y, m, rm) in enumerate(loader):
            b, y, m, rm = b.to(device), y.to(device), m.to(device), rm.to(device)
            chi = network(b)

            b = b * b_std + b_mean
            y = y * y_std + y_mean
            chi = chi * y_std + y_mean

            if writer:
                writer.add_scalar(f'{dataset_name} PSNR/Image {batch+1}', loss_psnr(y * m, chi * m), step)
                writer.add_scalar(f'{dataset_name} RMSE/Image {batch+1}', loss_rmse(y * m, chi * m), step)
                writer.add_scalar(f'{dataset_name} ddRMSE/Image {batch+1}', loss_ddrmse(y * m, chi * m), step)
                writer.add_scalar(f'{dataset_name} ROI_ddRMSE/Image {batch+1}', loss_ddrmse(y * rm, chi * rm), step)
                writer.add_scalar(f'{dataset_name} SSIM/Image {batch+1}', loss_ssim(y * m, chi * m), step)
                writer.add_scalar(f'{dataset_name} HFEN/Image {batch+1}', loss_hfen(y * m, chi * m), step)

                input_images = (torch.clamp(b[0, 0, :, :, ::20].permute(2, 0, 1).unsqueeze(1), min=-0.05, max=0.05) * 10) + 0.5
                output_images = (torch.clamp(chi[0, 0, :, :, ::20].permute(2, 0, 1).unsqueeze(1), min=-0.1, max=0.1) * 5) + 0.5
                label_images = (torch.clamp(y[0, 0, :, :, ::20].permute(2, 0, 1).unsqueeze(1), min=-0.1, max=0.1) * 5) + 0.5
                writer.add_image(f'{dataset_name} {batch+1}/Input', make_grid(input_images, padding=20), step)
                writer.add_image(f'{dataset_name} {batch+1}/Output', make_grid(output_images, padding=20), step)
                writer.add_image(f'{dataset_name} {batch+1}/Label', make_grid(label_images, padding=20), step)


def challenge_evaluation(name, network, writer, step):
    with torch.no_grad():
        print('Evaluating model on challenge dataset...')
        network.eval()
        for method in ['VSHARP', 'PDF']:
            inference(identifier, name, f'{identifier}-epoch{step:02d}.ckpt', method, called_in_train=True, network=network, writer=writer, step=step)


def train(load_path=None):
    # Define exp name
    exp_name = input("Experience Name (You may leave this blank): ")
    exp_name_ = datetime.now().strftime("%m-%d-%H-%M") + ('-' + exp_name) if exp_name else ''

    # Prepare status save location
    if not os.path.exists(save_base_path):
        os.mkdir(save_base_path)

    if not os.path.exists(os.path.join(save_base_path, identifier)):
        os.mkdir(os.path.join(save_base_path, identifier))

    save_path = os.path.join(save_base_path, identifier, exp_name_)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Save constants.py as json file
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(C, f, default=lambda x: str(x), indent=True)

    # Load dipole kernel
    d = load_dipole().to(device)

    # Prepare training and validation datasets
    train_dataset = QSMDataset('train')
    b_mean = train_dataset.X_mean
    b_std = train_dataset.X_std
    y_mean = train_dataset.Y_mean
    y_std = train_dataset.Y_std

    train_loader = DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(QSMDataset('val'), batch_size=1, shuffle=False)

    # Initialize training elements
    if load_path:
        checkpoint = torch.load(load_path, map_location='cpu')
        start_epoch = checkpoint.get('epoch')
        network = checkpoint.get('model')
        optimizer = checkpoint.get('optimizer')
        del checkpoint

        network = network.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        start_epoch = 0
        network = QSMNet().to(device)
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Initialize tensorboard logging
    writer = SummaryWriter(comment='_' + identifier + '_' + exp_name)
    writer.add_graph(network, torch.zeros(1, 1, 64, 64, 64, device=device))

    # Announce training
    print('{} training network "{}" for {} epoches on {}{}.'.format('Resume' if load_path else 'Begin', identifier,
                                                                    train_epochs, device.type,
                                                                    ':' + str(
                                                                        device.index) if device.index is not None else ''))

    # Begin training
    step = start_epoch * len(train_loader)  # Global step count for tensorboard
    for epoch in range(start_epoch, start_epoch + train_epochs):

        network.train()
        loss_history = []
        
        if epoch < one_to_two - 1:
            print("Loss gear phase one")
            total_loss.phase = 'one'
        else:
            print("Loss gear phase two")
            total_loss.phase = 'two'

        for batch, (b, y, m) in enumerate(train_loader):
            b, y, m = b.to(device), y.to(device), m.to(device)

            optimizer.zero_grad()

            chi = network(b)

            loss, loss_parts = total_loss(chi, y, b, d, m, b_mean, b_std, y_mean, y_std)
            loss.backward()
            step += 1
            optimizer.step()

            loss_cpu = loss.detach().cpu().item()
            loss_history.append(loss_cpu)

            torch.cuda.empty_cache()

            with torch.no_grad():
                # Track loss with tensorboard
                writer.add_scalar('Loss/Total loss', loss_cpu, step)
                for loss_name, loss_value in loss_parts.items():
                    writer.add_scalar('Loss/' + loss_name, loss_value, step)

                # Print step status
                if (batch + 1) % print_every == 0:
                    train_log = 'Epoch {:2d}/{:2d}\tLoss: {:.6f}\tTrain: [{}/{} ({:.0f}%)]        '.format(
                        epoch + 1, start_epoch + train_epochs, loss_cpu, batch + 1,
                        len(train_loader),
                        100. * batch / len(train_loader))
                    print(train_log, end='\r')
                    sys.stdout.flush()

        with torch.no_grad():
            # Print epoch status
            last_epoch = epoch + 1
            average_loss = sum(loss_history) / len(loss_history)
            epoch_end_log = "Epoch {:02d} completed, Average Loss is {:.6f}.            ".format(last_epoch, average_loss)
            print(epoch_end_log)

            # Save training status
            if last_epoch % save_every == 0:
                ckpt_save_path = os.path.join(save_path, '{}-epoch{:02d}.ckpt'.format(identifier, last_epoch))
                torch.save({
                    'epoch': last_epoch,
                    'model': network,
                    'optimizer': optimizer,
                }, ckpt_save_path)
                print("Saved Model on epoch {}.".format(last_epoch))

        # Evaluate model
        #evaluate_model('validation', network, val_loader, last_epoch, save_path, writer)
        challenge_evaluation(exp_name_, network, writer, last_epoch)

    writer.close()


if __name__ == '__main__':

    # Continue training from a saved training state
    if len(sys.argv) == 2:
        if not os.path.isfile(sys.argv[-1]):
            raise ValueError("File does not exist.")
        identifier = sys.argv[-1].split('/')[1]
        identifier = input(f"Identifier Name (Input nothing to use '{identifier}'): ") or identifier
        train(load_path=sys.argv[-1])
    # Start training from scratch
    elif len(sys.argv) == 1:
        identifier = input(f"Identifier Name (Input nothing to use '{identifier}'): ") or identifier
        train()
    else:
        raise ValueError('Multiple load paths detected. Did you forget the quotation marks?')
