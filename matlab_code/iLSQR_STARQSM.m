load('../../Data/training_before_patch.mat');
H=[0 0 1];
voxelsize=[1 1 1];
padsize=[12 12 12];
B0=3;
TE=25;

for i=1:7
    eval(sprintf('multiphs%d=multiphs%d(:,:,:,1:5);',i,i));
    eval(sprintf('tempPhase=multiphs%d;',i));
    eval(sprintf('tempCos=multicos%d;',i));
    eval(sprintf('load(''../../Data/Mask/mask%d.mat'');',i));
    tempMask=Mask;
    for j=1:5
        tempTissuePhase=2*pi*3*42.58*0.025*tempPhase(:,:,:,j);       
        tempStar(:,:,:,j)=QSM_star(tempTissuePhase,tempMask(:,:,:,j),'TE',TE,'B0',B0,'H',H,'padsize',padsize,'voxelsize',voxelsize);
        tempiLSQR(:,:,:,j)=QSM_iLSQR(tempTissuePhase,tempMask(:,:,:,j),'TE',TE,'B0',B0,'H',H,'padsize',padsize,'voxelsize',voxelsize);
    end
    eval(sprintf('starQSM%d=tempStar;',i));
    eval(sprintf('iLSQR%d=tempiLSQR;',i));
end

save('../../Data/starQSMbeforePatch.mat','-mat','starQSM*','-v7.3');
save('../../Data/iLSQRbeforePatch.mat','-mat','iLSQR*','-v7.3');


