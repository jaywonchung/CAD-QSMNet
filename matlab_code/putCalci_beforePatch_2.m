clear multi*
load('../../Data/training_set_before_patch.mat');
load('../../Data/trainingCalciMask.mat');
patientSet=[1,2,3,4,5,6,7];
displacementArray=[85,80,110];
howManyCalci=1;
linearity=2;
for patientNumber=patientSet
   eval(sprintf('multicos%d=multicos%d(:,:,:,1:5);',patientNumber,patientNumber)); 
end



for patientNumber=patientSet
   for aug=1:5
       mesg=sprintf('patient%d,aug %d',patientNumber,aug);
       disp(mesg);
       eval(sprintf('tempCos=linearity*multicos%d(:,:,:,%d);',patientNumber,aug));

       for numOfCalci=1:howManyCalci
           tempCalci=calciArray(:,:,:,numOfCalci+(aug-1)*howManyCalci+(patientNumber-1)*20);  
           tempMask=maskArray(:,:,:,numOfCalci+(aug-1)*howManyCalci+(patientNumber-1)*20);     
           [xx,yy,zz]=size(tempMask);
           x_0=displacementArray(1);
           y_0=displacementArray(2);
           z_0=displacementArray(3);
           for x=1:xx
               for y=1:yy
                   for z=1:zz
                       if tempMask(x,y,z)==1
                           tempCos(x_0-5+x,y_0-5+y,z_0-5+z)=tempCalci(x,y,z);
                       end
                   end
               end
           end
       end
       eval(sprintf('multicos%d(:,:,:,%d)=tempCos;',patientNumber,aug));
       
   end
    
end

save('putCalci_2.mat','-mat','multicos*','-v7.3');