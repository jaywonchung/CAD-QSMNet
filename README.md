# Introduction

Implementation of and experiments based on the QSMNet (https://arxiv.org/abs/1803.05627).  
Now CAD-QSMNet is not an implementation of the original QSMNet any more, but we indeed heavily drew inspiration from it.  
This work was done to submit to the [QSM Reconstruction Challenge 2.0](http://qsm.snu.ac.kr/?page_id=30).

# Contributions

Names are listed in alphabetical order:  
- Jaewon Chung ([@jaywonchung](https://github.com/jaywonchung))
- Jinho Park ([@jinh0park](https://github.com/jinh0park))
- Yunchan Hwang([@yunchfruit](https://github.com/yunchfruit))

Special thanks to Woojin Jung ([@wjjung93](https://github.com/wjjung93)) for his guidance and helpful advice during the challenge.

# Test your idea

## First Time Only
1. Clone this repository.
   ```bash
   git clone https://github.com/jinh0park/QSMFirstOrDeath.git
   ```
2. Install prerequisites.
    
    Using Anaconda3.
    torch >= 1.1.0
    
    - Cuda version == 8.0
   ```bash
   conda install pytorch=0.4.1 cuda80 -c pytorch
   pip install -r requirements.txt
   ```
   - Cuda version == 9.0
   ```bash
   conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
   pip install -r requirements.txt
   ```
   - Cuda version == 10.0
   ```bash
   conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
   pip3 install -r requirements.txt
   ```

## Already cloned?
1. Pull the latest changes from the repository.
   ```bash
   git pull
   ```
   
2. Create a new branch in the repository.
   ```bash
   git checkout -b your_branch_name
   ```

3. Change parameters in [```constants.py```](https://github.com/jinh0park/QSMFirstOrDeath/blob/master/constants.py).
   
4. Change the code directly.
   - [```data_utils.py```](https://github.com/jinh0park/QSMFirstOrDeath/blob/master/data_utils.py) defines how the code loads data from .mat files.
   - [```model.py```](https://github.com/jinh0park/QSMFirstOrDeath/blob/master/model.py) defines the QSMNet model layers and their connections.
   - [```train.py```](https://github.com/jinh0park/QSMFirstOrDeath/blob/master/train.py) defines the training logic. You can change the optimizer here.
   - [```loss.py```](https://github.com/jinh0park/QSMFirstOrDeath/blob/master/loss.py) defines the loss functions. ```train.py``` calls the ```total_loss``` function for the final loss.

5. Start training from scratch with
   ```bash
   python3 train.py
   ```
   Or, resume training from a saved training state with
   ```bash
   python3 train.py "path/to/saved/file"
   ```
   FYI, ```train.py``` saves the training state (epochs done, model weights, optimizer, and learning rate scheduler) every epoch (by default).

6. Check your training status with PyTorch TensorBoard.
   ```bash
   tensorboard --logdir runs --host localhost
   ```
   Once tensorboard runs, navigate to the given address (e.g. localhost:6006) on your machine's web browser.

7. When you're done, you may merge the branch to ```master``` if it should be part of the baseline model by sending a pull request. If not, you can delete the branch with
   ```bash
   git checkout master
   git branch -d your_branch_name
   ```
   But be careful; this will remove all the changes you made to that branch!


## Create inferences

```inference.py``` helps you perform inferences with the checkpoint file. It creates .mat and .nii files with inference results on the challenge data and the test set.

1. Requirements
   - The checkpoint file and its path in the form ```Checkpoints/<model_architecture>/<experience_name>/<model_name>.ckpt```.
   - The ```config.json``` file for the checkpoint, in the ```Checkpoints/<model_architecture>/<experience_name>``` folder.
   - The training data used to train the model. We need the input and output mean/std from it.
   - **Run ```bash ../QSMFirstOrDeath/prepare_inference.sh /path/to/NAS/list3``` in the ```Data``` folder. This will (hopefully) have you meet all the requirements below this line.**
   - In the ```../Data``` folder:
     - ```Challenge_stage1_sim1_fix.mat```
     - ```Challenge_stage1_sim2_fix.mat```
     - ```Challenge_stage1_ROI_mask.mat```
     - ```Test_0718.mat```
     - ```Test_ROI_mask.mat```
   - In the ```../Data/Conventional``` folder:
     - ```Sim1_Input.nii``` and ```Sim2_Input.nii```
     - ```Sim1_{method}.nii``` and ```Sim2_{method}.nii```, where ```method``` is the reconstruction method.
     - ```Test_{method}.mat``` where ```method``` is the reconstruction method.
     - Also check and edit the two lists named '```methods```' in ```inference.py``` appropriately. The first one is for the challenge inference, and the second one is for the test set inference.
2. Usage
   ```bash
   python inference.py 'Checkpoints/<model_architecture>/<experience_name>/<model_name>.ckpt'
   ```
3. A quick inference function: ```inf```
   - Import:   
        ```python
        from inference import inf
        ```
   - Arguments:
     - ```checkpoint_file```: path to the checkpoint file, relative to the QSMFirstOrDeath folder
     - ```file```: (optional) network input of extension ```.mat```, ```.nii``` or ```.nii.gz```
     - ```numpy```: (optional) network input of type ```np.ndarray```
     - ```tensor```: (optional) network input of type ```torch.Tensor```
     - ```device```: (optional) which device to use for inference. If not given, takes value from ```constants.py```.
   - Usage: Provide one, two, or all of ```file```, ```numpy```, or ```tensor```.
     - When only one input argument is given, ```inf``` returns its inference value. For ```file``` and ```numpy```, the output is a ```np.ndarray```. For ```tensor```, the output is a ```torch.Tensor```.
     - When more than one of the input arguments are given, ```inf``` returns a tuple of outputs. Inferences are made separately.
