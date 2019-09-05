clearvars
%------------------------------------------------------------
%------------------------customize--------------------------
%------------------------------------------------------------

%You can customize values below
%keep in mind that in _Addr value, you should put upto '.mat', not just a folder address
%Augementation: 5 or 10 usually
%patientSet: [1,2,3,4,5] when you want to use patient#1 to #5 as a training data
%Be careful when changing 'patientSet'. Make sure you have corresponding
%mask data in a mask .mat file.
%PATCHSIZE:8820=6(along X)*7(along Y)*6(along Z)*5(rotation)*7(patient 1 to 7)

DATA_NAME='../../Data/finetuning8700.mat'; %you can put a full path too
training_before_Addr='../../Data/finetuningBeforePatch.mat';
mask_total_UpToFolder='../../Data/Mask/';
Augmentation=5;
patientSet=[1,2,3,4,5,6,7];
%------------------------------------------------------------
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

%-------------------------DataTosave---------------------------------
DataToSave={'DATA_NAME','mask_t','input','label',...
    'input_std_total','input_mean_total','label_mean_total','label_std_total'};
%You can add or get rid of any name in 'DataToSave' but 'DATA_NAME'. 
%DO NOT delete DATA_NAME in 'DataToSave'


%DO NOT MODIFY 'DataNeeded'
DataNeeded={'DataNeeded','DataToSave'...
    'DATA_NAME','mask_total_UpToFolder','training_before_Addr','Augmentation','patientSet',...
    'mask_t','input','label',...
    'input_mean','input_std','label_mean','label_std',...
    'input_std_total','input_mean_total','input_std_total','label_mean_total','label_std_total'};
%----------------------------------------------------------------




%------------------------------------------------------------
%---------------Now We GET Statistical values----------------
%------------------------------------------------------------
disp('now we get statistical values this will take upto a minute')

load(training_before_Addr);


for patientNumber=patientSet
    eval(sprintf('tempMask=mask_%d;',patientNumber));

    eval(sprintf('data_cosmos = multicos%d;',patientNumber)); %equals to data1=multicos1,cos2,cos 3 and so on
    eval(sprintf('data_phase = multiphs%d;',patientNumber));
    
    zeroMask=zeros(size(tempMask));
    tempMask=zeroMask+tempMask;
    tempMask(tempMask==0)=NaN;
    data_cosmos=data_cosmos.*tempMask;
    data_phase=data_phase.*tempMask;
    
    allInputConcat(:,:,:,:,patientNumber)=data_phase;
    allLabelConcat(:,:,:,:,patientNumber)=data_cosmos;

    clear tempMask

    
end
input_std_total=nanstd(allInputConcat(:));
label_std_total=nanstd(allLabelConcat(:));
input_mean_total=nanmean(allInputConcat(:));
label_mean_total=nanmean(allLabelConcat(:));

clear allInputConcat
clear allLabelConcat
%------------------------------------------------------------
%---------------Now We've GOT Statistical values-------------
%------------------------------------------------------------
disp('statistical values done')





%------------------------------------------------------------
%---------------Now We GET mask patches----------------------
%------------------------------------------------------------
disp('making mask patches... this can take upto 2mins');


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
    eval(sprintf('data1=mask_%d;',patientNumber));
    for aug = 1:Augmentation
        maskAug=mod(aug-1,5)+1;
        data_L = data1(3:174,:,1:159,(maskAug));
        for jj=1:tt
             maskPatches(:,:,:,patientNumber,tt*(aug-1)+jj) = data_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
        end
    end
end
maskPatches = maskPatches(:,:,:,:);
mask_t(1,:,:,:,:) = single(maskPatches);
mask_t = permute(mask_t,[1 4 3 2 5]);
clear maskPatches
%------------------------------------------------------------
%---------------Now We've GOT mask patches-------------------
%------------------------------------------------------------
disp('making Mask patches Done')

clearvars('-except',DataNeeded{:})





%------------------------------------------------------------
%---------------Now We GET data patches----------------------
%------------------------------------------------------------
disp('making training patches...this can take upto several mins');

load(training_before_Addr);
%---------------input patches--------------------------------
disp('making training input patches...');
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
    eval(sprintf('data2 = multiphs%d;',patientNumber));
    for aug = 1:Augmentation 
        data_I = data2(3:174,:,1:159,(aug));%data_l is edge trimed phase data
        for jj=1:tt
            data_input(:,:,:,patientNumber,tt*(aug-1)+jj) = data_I(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
        end 
    end
end

 data_input = data_input(:,:,:,:); % patches just concatenated when they are from same patient
 input(1,:,:,:,:) = data_input; %for the sake of X = tf.placeholder("float", [None, PS, PS, PS, 1])
 clear data_input
 input = permute(input,[1 4 3 2 5]); %not 100% sure what is actaully going on, but needed for h5py to read it
save(DATA_NAME,'-mat','input','mask_t','input_std_total','input_mean_total','label_mean_total','label_std_total','-v7.3');
clear input mask_t
%---------------label patches--------------------------------
clearvars('-except',DataNeeded{:})
load(training_before_Addr);
 disp('making training label patches...');
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
    eval(sprintf('data1 = multicos%d;',patientNumber)); %equals to data1=multicos1,cos2,cos 3 and so on
    for aug = 1:Augmentation 
        data_L = data1(3:174,:,1:159,(aug));%data_L is edge trimmed susceptibility
        for jj=1:tt
            data_label(:,:,:,patientNumber,tt*(aug-1)+jj) = data_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
        end 
    end
end
data_label = data_label(:,:,:,:); % patches just concatenated when they are from same patient
label(1,:,:,:,:) = data_label; 
clear data_label
label = permute(label,[1 4 3 2 5]);

%------------------------------------------------------------
%---------------Now We've GOT data patches-------------------
%------------------------------------------------------------

disp('making data paches done');






%------------------------------------------------------------
%---------------Now We save Data-----------------------------
%------------------------------------------------------------
disp('Saving... this takes around 10mins for 8400 patches.')
clearvars('-except',DataToSave{:})

save(DATA_NAME,'-append','label');


disp('All Done')
