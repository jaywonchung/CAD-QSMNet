clearvars

%---------------------------customize section-----------

DATA_NAME='../../Data/Train_0712_8400_model_UptoThree_conti.mat';
training_before_Address='../../Data/training_set_before_patch.mat';
mask_total_UpToFolder='../../Data/Mask/';
Upto=3;
meanScale=(1+Upto)/2;
stdScale=meanScale;
patientSet=[1,2,3,4,5,6,7];
matrix_size=[176,176,160];


%-------------------------------------------------------
voxel_size=[1,1,1];
B0_dir=[0,0,1];
D=dipole_kernel(matrix_size,voxel_size,B0_dir);

load(training_before_Address);

for i =patientSet
    eval(sprintf('phase_%d=multiphs%d;',i,i));
    eval(sprintf('cosmos_%d=multicos%d;',i,i));
end
clear multi*

for patientNumber=patientSet
    currentMask=sprintf('mask%d.mat',patientNumber);
    currentMask=strcat(mask_total_UpToFolder,currentMask);
    load(currentMask);
    mask_temp=Mask;
    
    eval(sprintf('mask_%d=mask_temp;',patientNumber));
    clear mask_temp
    clear Mask;
end

for patientNumber=patientSet
   eval(sprintf('phase_%d=phase_%d(:,:,:,1:5);',patientNumber,patientNumber));
   eval(sprintf('cosmos_%d=cosmos_%d(:,:,:,1:5);',patientNumber,patientNumber));
    
end


for patientNumber=patientSet
    
    eval(sprintf('current_cosmos=cosmos_%d;',patientNumber));
    eval(sprintf('current_phase=phase_%d;',patientNumber));
    for i=1:5
        current_phase(:,:,:,i)=ifftn(fftn(current_cosmos(:,:,:,i)).*D);
    end
    
    eval(sprintf('cosmos_%d=current_cosmos;',patientNumber));
    eval(sprintf('phase_%d=current_phase;',patientNumber));
    
end
disp('Loading mask and training data on workspace done')

for patientNumber=patientSet

    eval(sprintf('data_cosmos = cosmos_%d;',patientNumber)); %equals to data1=multicos1,cos2,cos 3 and so on
    eval(sprintf('data_phase = phase_%d;',patientNumber));
    eval(sprintf('data_mask=mask_%d;',patientNumber));
        
    zeroMask=zeros(size(data_mask));
    data_mask=zeroMask+data_mask;
    data_mask(data_mask==0)=NaN;
    data_cosmos=data_cosmos.*data_mask;
    data_phase=data_phase.*data_mask;
    
    allInputConcat(:,:,:,:,patientNumber)=data_phase;
    allLabelConcat(:,:,:,:,patientNumber)=data_cosmos;

    clear Mask
end
input_std_total=stdScale*nanstd(allInputConcat(:));
label_std_total=stdScale*nanstd(allLabelConcat(:));
input_mean_total=meanScale*nanmean(allInputConcat(:));
label_mean_total=meanScale*nanmean(allLabelConcat(:));
clear allInputConcat
clear allLabelConcat
save(DATA_NAME,'-mat','input_std_total','input_mean_total','label_mean_total','label_std_total','-v7.3');


%------------------------------------------------------------
%---------------Now We GET mask patches----------------------
%------------------------------------------------------------
disp('making and saving mask patches...');
PseudoIterator={'PP','PP_y','SS','PS','str_x','str_y','str_z',...
    'yy','xx','kk','zz','tt'};
    

%just making pseudo iterator to facilitate patching process
PP = 7; PP_y = 8; SS = 6; PS = 64; %how many slices along x,y,z and patch size 64 
str_x = (172-PS)/(PP-1); %stride along x
str_y = (176-PS)/(PP_y-1); %stride along y
str_z = (159-PS)/(SS-1); %stride along z
[yy, xx] = meshgrid(1:PP_y,1:PP); xx = repmat(xx,[1,1,SS]); yy = repmat(yy,[1,1,SS]);
for kk=1:SS
    zz(:,:,kk) = ones(PP,PP_y)*kk;
end
xx=xx(:); yy=yy(:); zz=zz(:);  tt=length(xx);
%making pseudo iterator done


for patientNumber=patientSet
    eval(sprintf('currentMask=mask_%d;',patientNumber));
    eval(sprintf('currentPhase = phase_%d;',patientNumber));
    eval(sprintf('currentCosmos = cosmos_%d;',patientNumber));

    for aug = 1:5
        tempMask = currentMask(3:174,:,1:159,aug);
        tempCosmos=currentCosmos(3:174,:,1:159,aug);
        tempPhase=currentPhase(3:174,:,1:159,aug);
        for jj=1:tt
            scaleFactor=1+rand*(Upto-1);
             maskPatches(:,:,:,patientNumber,tt*(aug-1)+jj) = tempMask(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
             inputPatches(:,:,:,patientNumber,tt*(aug-1)+jj) = scaleFactor*tempPhase(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
             labelPatches(:,:,:,patientNumber,tt*(aug-1)+jj) = scaleFactor*tempCosmos(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
            
        end
    end
end
clearvars(PseudoIterator{:});



maskPatches = maskPatches(:,:,:,:);
mask_t(1,:,:,:,:) = single(maskPatches);
clear maskPatches
inputPatches=inputPatches(:,:,:,:);
input(1,:,:,:,:)=inputPatches;
clear inputPatches
labelPatches=labelPatches(:,:,:,:);
label(1,:,:,:,:)=labelPatches;
clear labelPatches

mask_t = permute(mask_t,[1 4 3 2 5]);
input=permute(input,[1 4 3 2 5]);
label=permute(label,[1 4 3 2 5]);

save(DATA_NAME,'-append','mask_t','input','label');

disp('saving Mask patches Done')






