clearvars

DATA_NAME='../../Data/finetuningBeforePatch.mat'; %you can put a full path too
training_before_Addr='../../Data/training_set_before_patch.mat';
mask_total_UpToFolder='../../Data/Mask/';
Augmentation=5;
patientSet=[1,2,3,4,5,6,7];
%----------------------------------------------------------
%------------------------------------------------------------

noiseConstant=0.01
noiseMax=0.05;
noiseMin=0.01;
noiseNumber=35;
noiseDelta=(noiseMax-noiseMin)/noiseNumber;
noiseArray=noiseMin:noiseDelta:noiseMax;

%------------------------------------------------------------
%------------------------------------------------------------

for patientNumber=1:7
        
    currentMask=sprintf('mask%d.mat',patientNumber);
    currentMask=strcat(mask_total_UpToFolder,currentMask);
    load(currentMask);
    mask_temp=Mask;
    
    eval(sprintf('mask_%d=mask_temp;',patientNumber));
    clear mask_temp
    clear Mask;
end

load(training_before_Addr);

for i= 1:7
    
    eval(sprintf('tempMask=mask_%d;',i));
    eval(sprintf('tempCosmos=multicos%d;',i));
    eval(sprintf('tempPhase=multiphs%d;',i));
    for aug=1:5
       tempNoiseConstant=noiseArray(aug+(i-1)*5);
       
       tempNoise=tempNoiseConstant*randn(size(tempCosmos(:,:,:,1)));
       tempPhase(:,:,:,aug)=tempMask(:,:,:,aug).*tempNoise+tempCosmos(:,:,:,aug);
    end
    eval(sprintf('multiphs%d=tempPhase;',i));    
    
end
%%
save(DATA_NAME,'-mat','multi*','-v7.3');



 