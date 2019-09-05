clearvars
training_before_Address='../../Data/training_set_before_patch.mat';
mask_total_UpToFolder='../../Data/Mask/';
DATA_NAME='../../Data/TKD_before_patch.mat'
matrix_size=[176,176,160];

load(training_before_Address);

%%
for patientNumber=1:7
    currentMask=sprintf('mask%d.mat',patientNumber);
    currentMask=strcat(mask_total_UpToFolder,currentMask);
    load(currentMask);
    mask_temp=Mask;
    
    eval(sprintf('mask_%d=mask_temp;',patientNumber));
    clear mask_temp
    clear Mask;
end
%%
voxel_size=[1,1,1];
B0_dir=[0,0,1];
D=dipole_kernel(matrix_size,voxel_size,B0_dir);
D_binary_plus=D>=0;
D_binary_minus=D<0;
D_binary_minus=-1*D_binary_minus;
D_binary=D_binary_plus+D_binary_minus;
third_D=D_binary/3;

x=ifftn(fftn(multiphs1(:,:,:,3))./third_D);
for i=1:7
    eval(sprintf('multicos%d=multicos%d(:,:,:,1:5);',i,i));
    eval(sprintf('multiphs%d=multiphs%d(:,:,:,1:5);',i,i));
end

for i=1:7
   eval(sprintf('tempMask=mask_%d;',i));
   eval(sprintf('tempPhase=multiphs%d;',i));
   for j=1:5
      tempTKD(:,:,:,j)=ifftn(fftn(tempPhase(:,:,:,j))./third_D);
      tempTKD(:,:,:,j)=tempMask(:,:,:,j).*tempTKD(:,:,:,j);
   end
   eval(sprintf('multicos%d=multicos%d-tempTKD;',i,i));
end
%%
save(DATA_NAME,'-mat','multicos*','multiphs*','-v7.3');

