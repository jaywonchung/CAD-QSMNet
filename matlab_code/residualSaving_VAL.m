
load('../../Data/val_phscos.mat');
load('../../Data/starQSMbeforePatch_val.mat');
%%
for i=8:8
    eval(sprintf('multiphs%d=multiphs%d(:,:,:,1:5);',i,i));
    eval(sprintf('multicos%d=multicos%d(:,:,:,1:5);',i,i));
    eval(sprintf('multires%d=multicos%d(:,:,:,1:5)-starQSM%d;',i,i,i));
    eval(sprintf('multiconven%d=starQSM%d;',i,i));
end
save('../../Data/starResidual_before_patch_val.mat','-mat','multicos*','multiphs*','multires*','multiconven*','-v7.3');


%%
clearvars
load('../../Data/val_phscos.mat');
load('../../Data/iLSQRbeforePatch_val.mat');
for i=8:8
    eval(sprintf('multiphs%d=multiphs%d(:,:,:,1:5);',i,i));
    eval(sprintf('multicos%d=multicos%d(:,:,:,1:5);',i,i));
    eval(sprintf('multires%d=multicos%d(:,:,:,1:5)-iLSQR%d;',i,i,i));
    eval(sprintf('multiconven%d=iLSQR%d;',i,i));
    
end
save('../../Data/iLSQRResidual_before_patch_val.mat','-mat','multicos*','multiphs*','multires*','multiconven*','-v7.3');

