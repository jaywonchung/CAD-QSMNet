clearvars

load('../../Data/Test_ROI_mask.mat');
roi_mask=permute(roi_mask,[4 3 2 5 1]);
roi_mask=roi_mask(:,:,:,17:32);

load('../../Data//Conventional/Test_QSMNet.mat');
QSMNet=permute(QSMNet,[4 3 2 5 1]);
QSMNet=QSMNet(:,:,:,5);

NAS_Addr_UpTo_PersonalFolder='/data/list3/Personal_folder'
patient10='/woojin/QSM_challenge/Quantitative_metric/Testset_ROI/test_subject4/phscos.mat';
patient10Address=strcat(NAS_Addr_UpTo_PersonalFolder,patient10);
load(patient10Address);
label=multiphs10(:,:,:,1);
clear multi*
%------------------------------------------------------%
checkpointFolder   = '../Checkpoints/';

FileList = dir(fullfile(checkpointFolder, '**', 'Test_inference_label.mat'));
for folderNum=1:length(FileList)
    mesg=sprintf('working on %d/%d...',folderNum,length(FileList));
    disp(mesg);
    
    currentAddress=FileList(folderNum).folder;
    where=split(currentAddress,'/');
    whereAbout=strcat(where(end-3),'/',where(end-2),'/',where(end-1));
    disp(whereAbout);
    ours=getOurs(currentAddress);

    for i=1:16
        tempOurs=getOnlyROI(ours,roi_mask(:,:,:,i));
        tempLabel=getOnlyROI(label,roi_mask(:,:,:,i));
        tempQSMNet=getOnlyROI(QSMNet,roi_mask(:,:,:,i));
        qsmDdRMSE(i)=ddRMSE(tempQSMNet,tempLabel);
        oursDdRMSE(i)=ddRMSE(tempOurs,tempLabel);
    end
    saveAddress=strcat(currentAddress,'/ddRMSEratio.png');
    saveas(makeFigure(qsmDdRMSE,oursDdRMSE),saveAddress);  
    
    
end

function y=getOurs(address)
loadAddress=strcat(address,'/Test_inference_label.mat');
load(loadAddress);
y=chi10Real;

end


function y=getOnlyROI(matrix,roimask)
newMatrix=matrix.*roimask;
newMatrix=newMatrix(:);
newMatrix(roimask==0)=NaN;
newMatrix=newMatrix(~isnan(newMatrix))';
y=newMatrix;
end

function y=ddRMSE(out,label)
out=out(:);
label=label(:);
sumxy=sum(out.*label);
Xavg=mean(out);
Yavg=mean(label);
N=length(out);
X_Xavg=out-Xavg;
beta=(sumxy-N*Xavg*Yavg)/sum(X_Xavg.*X_Xavg);
alpha=Yavg-beta*Xavg;
delta=label-alpha-out*beta;
deltaSquare=delta.*delta;
deltaSquare=deltaSquare/(beta*beta+1);
% figure;
% plot(out,out*beta+alpha,'r');
% hold on
% scatter(out,label,'.','b');
y=sqrt(mean(deltaSquare(:)));
end

function y=makeFigure(qsmnet,ours)
y=figure('Visible','off','Units','inches','Position',[0 0 7 2]);
ratio=ours./qsmnet;
bar(ratio);
hold on
plot(xlim,[1 1],'black','LineStyle',':','LineWidth',1);
hold on
text(1:length(ratio),double(ratio),num2str(ratio', '%0.3f'),'vert','bottom','horiz','center','FontSize',7.5);
ylim([0.7 1.3]);

end
