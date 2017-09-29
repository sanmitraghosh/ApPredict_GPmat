
%%%%%%%%%%%%% This script runs the iterative active learning for
%%%%%%%%%%%%% accumulating training points for the O'hara model. NB: the
%%%%%%%%%%%%% dataset contains pretrained hyperparameters of a GP
%%%%%%%%%%%%% classifier and the APD90 dataset 0f 100K

close all
clear all
load('Alearning_4D_hyp.mat')

tic
CScale=1100;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% ITERATIVE ACTIVE LEARNING
%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cT=[2000,3000,5000,7000,10000,15000,20000];
counter=0;
    nu = fix(CScale); cu = randperm(length(gk)); cu = cu(1:nu);    
    Gk = gk(cu,:);
    APDkn=APDtrue(cu,:); %%% Start with random 1000 points
    APDtrain=APDkn; %%% these will be grown intelligently
    Gktrain=Gk; %%% same for these
    [E1,E2,E3,E4] = ndgrid(linspace(0,1,100));
    E=[E1(:),E2(:),E3(:),E4(:)]; %% create grid
    while length(Gk)<cT(5) %%% Now iterate until you accumulate desired no. of training pnts

        method='grid'; % Flag to trigger the swarming types or grid ('grid' 'pso' 'ga')
        tic
        [Xnew,APDnew,Enew] = sequentialGridpredictPSO( E,Gk,APDkn,hyp1, hyp2, hyp3 , method);
        toc;
        APDtrain=[APDtrain;APDnew];
        Gktrain=[Gktrain;Xnew];
%         APDkn=[APD;APDnew]; % this is when we use all points grown so far
%         Gk=[Gk;Xnew];
        APDkn=APDnew;
        Gk=Xnew; %% reuse only the last AScale number of good points for next predictn in iteratn
        tu=randperm(length(APDkn));
        Gk=Gk(tu,:);
        APDkn=APDkn(tu); %% Shuffle them
        E=Enew;
        counter=counter+1000
        disp(counter)
    end

parfor i=1:length(cT)
    %%%%%%%%  Classification test
    %%%%%%%%  %%%%%%%%%%%%%%%%%
    
    
    [Yt(:,i),Yp(:,i),~,lpNR(:,i),lpD(:,i),lpND(:,i),s1(:,i),s2(:,i),s3(:,i)]=TestScatterMulti([100,100],gk,APDtrue,1,1,Gktrain(1:cT(i),:),APDtrain(1:cT(i),:),hyp1,hyp2,hyp3);
end
toc;
MisClass=zeros(length(cT),1);

for j=1:length(cT)
for i=1:100000
    if Yt(i,j)~=Yp(i,j)
        MisClass(j)=MisClass(j)+1;
    end
end
end
plot(cT,100-MisClass/1000)%% Plot classification accuracy











% plot3(gk(:,1),gk(:,2),APDtrue,'*','MarkerSize',1)    
