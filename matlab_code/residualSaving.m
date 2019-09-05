
load('../../Data/training_before_patch.mat');
load('../../Data/starQSMbeforePatch.mat');
%%
for i=1:7
    eval(sprintf('multiphs%d=multiphs%d(:,:,:,1:5);',i,i));
    eval(sprintf('multicos%d=multicos%d(:,:,:,1:5)-starQSM%d;',i,i,i));
    
end
save('../../Data/starResidual_before_patch.mat','-mat','multicos*','multiphs*','-v7.3');


%%
clearvars
load('../../Data/training_before_patch.mat');
load('../../Data/iLSQRbeforePatch.mat');
for i=1:7
    eval(sprintf('multiphs%d=multiphs%d(:,:,:,1:5);',i,i));
    eval(sprintf('multicos%d=multicos%d(:,:,:,1:5)-iLSQR%d;',i,i,i));
    
end
save('../../Data/iLSQRResidual_before_patch.mat','-mat','multicos*','multiphs*','-v7.3');

