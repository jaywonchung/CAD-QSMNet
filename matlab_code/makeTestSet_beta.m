clearvars
% this saves patient data of 9 to 12.
%it fetches data from the NAS. So make sure that your desktop is connected
%to the NAS.
Augmentation=5;
DATA_NAME='somethingupto.mat'; %you can put a full path too
NAS_Addr_UpToTestset_ROI='/media/mynas/Personal_folder/woojin/QSM_challenge/Quantitative_metric/Testset_ROI/';

DataToSave={'DATA_NAME','mask_t','input','label',...
    'input_std_total','input_mean_total','label_mean_total','label_std_total'};


%load multicos9to12, multiphs 9to12 at once-------------------------%

for testSetNumber=3:6
    x=sprintf('test_subject%d/phscos.mat',testSetNumber);
    loadAddress=strcat(NAS_Addr_UpToTestset_ROI,x);
    load(loadAddress);

end

disp('loadingDone');


%-------------------------save Mask--------------------------------%
for patientNumber=1:4 % there are 4 patients used in the test set,test_subject 3 to 6
    %call dataOfTestPatientDataSet
    x=sprintf('test_subject%d/mask_t%d.mat',(patientNumber+2),(patientNumber+2));
    currentMask=strcat(NAS_Addr_UpToTestset_ROI,x);
    load(currentMask);
    mask_1=Mask;
    for rotation=1:5 %there are 5 datasets of different roation(algined) for each patient
        mask_total(:,:,:,patientNumber,rotation)=mask_1(:,:,:,rotation);
    end
end
mask_total=mask_total(:,:,:,:);
maskData_stat=mask_total;
mask_t(1,:,:,:,:)=mask_total;
mask_t=permute(mask_t,[1 4 3 2 5]);
clear mask_total;

disp('maskDone');


%---------------------------save Input---------------------------%
for patientNumber=1:4 % there are 4 patients used in the test set///test_subject 3 to 6

    eval(sprintf('phase_singlePatient=multiphs%d;',(patientNumber+8)));
    for rotation=1:5 %there are 5 datasets of different roation(algined) for each patient
        testInput(:,:,:,patientNumber,rotation)=phase_singlePatient(:,:,:,rotation);
    end
end
%---------------------------save Label------------------------------%
for patientNumber=1:4 % there are 4 patients used in the test set,test_subject 3 to 6
    %call dataOfTestPatientDataSet
    eval(sprintf('susceptibility_singlePatient=multicos%d;',(patientNumber+8)));
    for rotation=1:5 %there are 5 datasets of different roation(algined) for each patient
        testLabel(:,:,:,patientNumber,rotation)=susceptibility_singlePatient(:,:,:,rotation);
    end
end
clear multiphs*
clear multicos*

testInput=testInput(:,:,:,:); % dataset just concatenated when they are from same patient
testLabel=testLabel(:,:,:,:); % dataset just concatenated when they are from same patient

inputData_stat=testInput;
labelData_stat=testLabel;

input(1,:,:,:,:)=testInput; %for the sake of X = tf.placeholder("float", [None, PS, PS, PS, 1])
clear testInput
input=permute(input,[1 4 3 2 5]); %for the sake of reverse ordering in h5py module

label(1,:,:,:,:)=testLabel; %for the sake of X = tf.placeholder("float", [None, PS, PS, PS, 1])
clear testLabel
label=permute(label,[1 4 3 2 5]); %for the sake of reverse ordering in h5py module


%statistical value
zeroMask=zeros(size(maskData_stat));
maskData_stat=zeroMask+maskData_stat;
maskData_stat(maskData_stat==0)=NaN;
labelData_stat=labelData_stat.*maskData_stat;
inputData_stat=inputData_stat.*maskData_stat;

input_std_total=nanstd(inputData_stat(:));
label_std_total=nanstd(labelData_stat(:));
input_mean_total=nanmean(inputData_stat(:));
label_mean_total=nanmean(labelData_stat(:));


%----------------------saving in .mat file-------------------------------%


disp('test input and label done...now saving');
clearvars('-except',DataToSave{:})

save(DATA_NAME,'-mat','-v7.3');
disp('alldone')