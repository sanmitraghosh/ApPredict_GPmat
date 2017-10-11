%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APD evaluate at scale on HPC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%

clear
startup
[t1,t2] = ndgrid(linspace(0,1,100)); 
gk = [t1(:),t2(:)]; 
gk(:,3)=1;gk(:,4)=1;
tic
[APDtrue]=EvaluateAPD(gk,100);
toc

save('Alearning_2D_10k_Grid.mat');
clear 
gk=unifrnd(0,1,4,100000)';
% gk(:,3)=1;gk(:,4)=1;
tic
[APDtrue]=EvaluateAPD(gk,100);
toc
save('Alearning_4D_100k_Test.mat');
exit;

% steps=linspace(0,100000,100000)';
% throw_chaste=[gk APDtrue];
% % % % dlmwrite('myParam.dat',throw_chaste,'Delimiter',' ','precision','%1.8e','coffset',2);
% fileID=fopen('matlab.txt','w');
% fprintf(fileID,'  %1.8e   %1.8e   %1.8e   %1.8e   %1.8e \n',throw_chaste');
% fclose(fileID);
% % exit;
% 
% plot3(gk(:,1),gk(:,2),APDtrue,'b*','MarkerSize',1);
