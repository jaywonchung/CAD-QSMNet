clearvars

disp('makingValidation .mat file')
%this is for making validation Data set. It has masks,phases,
%susceptibilities of patient 8. (5 rotation(augmentation) for each)
%it fetches data from the NAS. So make sure that your desktop is connected
%to the NAS.
%modify mask_total_Addr and patient8_Addr accordingly.

Augmentation=5;
DATA_NAME='somethingupto3.mat'; %you can put a full path too
%%%%CAUTION!!!! YOUR 'mask_total_Addr' must have corresponding mask data in it
mask_total_Addr='/media/mynas/Personal_folder/woojin/QSM_challenge/Quantitative_metric/Testset_ROI/test_subject2/mask_t2.mat';
patient8_Addr='/media/mynas/Personal_folder/woojin/QSM_challenge/Quantitative_metric/Testset_ROI/test_subject2/phscos.mat'; 


DataToSave={'DATA_NAME','mask_t','input','label',...
    'input_std_total','input_mean_total','label_mean_total','label_std_total'};
load(mask_total_Addr);
load(patient8_Addr);


%saving statistical value
mask_stat=Mask;
data_cosmos_stat=multicos8(:,:,:,1:5);
data_phase_stat=multiphs8(:,:,:,1:5);

zeroMask=zeros(size(mask_stat));
mask_stat=zeroMask+mask_stat;
mask_stat(mask_stat==0)=NaN;
data_cosmos_stat=data_cosmos_stat.*mask_stat;
data_phase_stat=data_phase_stat.*mask_stat;

input_std_total=nanstd(data_phase_stat(:));
label_std_total=nanstd(data_cosmos_stat(:));
input_mean_total=nanmean(data_phase_stat(:));
label_mean_total=nanmean(data_cosmos_stat(:));

%saveMask
mask_singlePatient=Mask;
for rotation=1:Augmentation %there are 5 datasets of different roation(algined) for each patient
    maskInput(:,:,:,rotation)=mask_singlePatient(:,:,:,rotation);
end


disp('maskDone');

phase_singlePatient=multiphs8;
for rotation=1:Augmentation %there are 5 datasets of different roation(algined) for each patient
    validationInput(:,:,:,rotation)=phase_singlePatient(:,:,:,rotation);
end



 % there are 5 patients used in the test set,test_subject 2 to 6
%refer to /media/mynas/Personal_folder/woojin/QSM_challenge/Testset_ROI
%call dataOfTestPatientDataSet
susceptibility_singlePatient=multicos8;
for rotation=1:Augmentation %there are 5 datasets of different roation(algined) for each patient
    validationLabel(:,:,:,rotation)=susceptibility_singlePatient(:,:,:,rotation);
end

validationInput=validationInput(:,:,:,:); % dataset just concatenated when they are from same patient
input(1,:,:,:,:)=validationInput; %for the sake of X = tf.placeholder("float", [None, PS, PS, PS, 1])
clear validationInput
input=permute(input,[1 4 3 2 5]); %for the sake of reverse ordering in h5py module

validationLabel=validationLabel(:,:,:,:); % dataset just concatenated when they are from same patient
label(1,:,:,:,:)=validationLabel; %for the sake of X = tf.placeholder("float", [None, PS, PS, PS, 1])
clear validationLabel
label=permute(label,[1 4 3 2 5]); %for the sake of reverse ordering in h5py module

maskInput=maskInput(:,:,:,:);
mask_t(1,:,:,:,:)=maskInput;
clear maskInput
mask_t=permute(mask_t,[1 4 3 2 5]);



clearvars('-except',DataToSave{:})

save(DATA_NAME,'-mat','-v7.3');
disp('alldone')