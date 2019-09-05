clearvars

%---------------------------customize section-----------
DATA_NAME='../../Data/calci_patch17640.mat';
calci_beforeAddress='../../Data/Calci.mat';
training_before_Address='../../Data/training_set_before_patch.mat';
mask_total_UpToFolder='../../Data/Mask/';

patientSet=[1,2,3,4,5,6,7];
matrix_size=[176,176,160];


%-------------------------------------------------------
voxel_size=[1,1,1];
B0_dir=[0,0,1];
D=dipole_kernel(matrix_size,voxel_size,B0_dir);

load(training_before_Address);

for i =1:7
    eval(sprintf('phase_%d=multiphs%d(:,:,:,1:5);',i,i));
    eval(sprintf('cosmos_%d=multicos%d(:,:,:,1:5);',i,i));
end
clear multi*

load(calci_beforeAddress);
for i=1:7
   eval(sprintf('cosmos_%d(:,:,:,6:10)=multicos%d;',i,i)); 
end

for patientNumber=1:7
    currentMask=sprintf('mask%d.mat',patientNumber);
    currentMask=strcat(mask_total_UpToFolder,currentMask);
    load(currentMask);
    mask_temp=Mask;
    for i =1:5
        mask_temp(:,:,:,i+5)=Mask(:,:,:,i);
    end
    clear Mask
    eval(sprintf('mask_%d=mask_temp;',patientNumber));
    clear mask_temp
end


for patientNumber=patientSet
    eval(sprintf('current_cosmos=cosmos_%d;',patientNumber));
    eval(sprintf('current_phase=phase_%d;',patientNumber));
    for i=1:5
        %current_cosmos(:,:,:,i+5)=lambda*current_cosmos(:,:,:,i);
        current_phase(:,:,:,i+5)=ifftn(fftn(current_cosmos(:,:,:,i+5)).*D);
    end
    eval(sprintf('cosmos_%d=current_cosmos;',patientNumber));
    eval(sprintf('phase_%d=current_phase;',patientNumber));
end
disp('Loading mask and training data on workspace done')
%%
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
input_std_total=nanstd(allInputConcat(:));
label_std_total=nanstd(allLabelConcat(:));
input_mean_total=nanmean(allInputConcat(:));
label_mean_total=nanmean(allLabelConcat(:));
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
PP = 6; PP_y = 7; SS = 6; PS = 64; %how many slices along x,y,z and patch size 64 
str_x = (174-PS)/(PP-1); %stride along x
str_y = (172-PS)/(PP_y-1); %stride along y
str_z = (159-PS)/(SS-1); %stride along z
[yy, xx] = meshgrid(1:PP_y,1:PP); xx = repmat(xx,[1,1,SS]); yy = repmat(yy,[1,1,SS]);
for kk=1:SS
    zz(:,:,kk) = ones(PP,PP_y)*kk;
end
xx=xx(:); yy=yy(:); zz=zz(:);  tt=length(xx);
%making pseudo iterator done


for patientNumber=patientSet
    eval(sprintf('data1=mask_%d;',patientNumber));
    for aug = 1:10
        data_L = data1(2:175,3:174,1:159,(aug));
        for jj=1:tt
             maskPatches(:,:,:,patientNumber,tt*(aug-1)+jj) = data_L(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
        end
    end
end

maskPatches = maskPatches(:,:,:,:);
mask_t(1,:,:,:,:) = single(maskPatches);
clear maskPatches

mask_t = permute(mask_t,[1 4 3 2 5]);
clearvars(PseudoIterator{:});

save(DATA_NAME,'-append','mask_t');
clear mask_t
disp('saving Mask patches Done')
%------------------------------------------------------------
%---------------Now We've GOT mask patches-------------------
%------------------------------------------------------------


%------------------------------------------------------------
%---------------Now We get input patches-------------------
%------------------------------------------------------------


disp('making training input patches...');
%just making pseudo iterator to facilitate patching process
PP = 6; PP_y = 7; SS = 6; PS = 64; %how many slices along x,y,z and patch size 64 
str_x = (174-PS)/(PP-1); %stride along x
str_y = (172-PS)/(PP_y-1); %stride along y
str_z = (159-PS)/(SS-1); %stride along z
[yy, xx] = meshgrid(1:PP_y,1:PP); xx = repmat(xx,[1,1,SS]); yy = repmat(yy,[1,1,SS]);
for kk=1:SS
    zz(:,:,kk) = ones(PP,PP_y)*kk;
end
xx=xx(:); yy=yy(:); zz=zz(:);  tt=length(xx);
%making pseudo iterator done

for patientNumber=patientSet
    eval(sprintf('data2 = phase_%d;',patientNumber));
    for aug = 1:10 
        data_I = data2(2:175,3:174,1:159,(aug));%data_l is edge trimed phase data
        for jj=1:tt
            data_input(:,:,:,patientNumber,tt*(aug-1)+jj) = data_I(str_x*xx(jj)-(str_x-1):str_x*xx(jj)+(PS-str_x),...
                str_y*yy(jj)-(str_y-1):str_y*yy(jj)+(PS-str_y),str_z*zz(jj)-(str_z-1):str_z*zz(jj)+(PS-str_z));
        end 
    end
end

data_input = data_input(:,:,:,:); % patches just concatenated when they are from same patient
input(1,:,:,:,:) = data_input; %for the sake of X = tf.placeholder("float", [None, PS, PS, PS, 1])
clear data_input

input = permute(input,[1 4 3 2 5]);
clearvars(PseudoIterator{:});

save(DATA_NAME,'-append','input');
clear input
disp('saving input patches Done')

%-------------------------------------------------------------------%
%------------------------------------------------------------
%---------------Now We get label patches-------------------
%------------------------------------------------------------


disp('making training label patches...');

%just making pseudo iterator to facilitate patching process
PP = 6; PP_y = 7; SS = 6; PS = 64; %how many slices along x,y,z and patch size 64 
str_x = (174-PS)/(PP-1); %stride along x
str_y = (172-PS)/(PP_y-1); %stride along y
str_z = (159-PS)/(SS-1); %stride along z
[yy, xx] = meshgrid(1:PP_y,1:PP); xx = repmat(xx,[1,1,SS]); yy = repmat(yy,[1,1,SS]);
for kk=1:SS
    zz(:,:,kk) = ones(PP,PP_y)*kk;
end
xx=xx(:); yy=yy(:); zz=zz(:);  tt=length(xx);
%making pseudo iterator done

for patientNumber=patientSet
    eval(sprintf('data1 = cosmos_%d;',patientNumber)); %equals to data1=multicos1,cos2,cos 3 and so on
    for aug = 1:10
        data_L = data1(2:175,3:174,1:159,(aug));%data_L is edge trimmed susceptibility
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
clearvars(PseudoIterator{:});

save(DATA_NAME,'-append','label');
clear label
disp('saving label patches Done')






