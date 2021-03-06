function [ x_train, y_train ] = intelligentTraining( x_InitTrain, y_InitTrain, gpoptions )

        counter=0;
        time=1;
        gpoptions.Batch=0;
        LM=gpoptions.LearningMode;

        E=unifrnd(0,1,1e4,size(x_InitTrain,2));
        
        if strcmp(LM,'surface')
            if gpoptions.surfAlClass==1
                 lab=justClassifier(x_InitTrain, y_InitTrain, E, gpoptions.classHyperParams ) ; 
                 Cord=find(lab==1);
                 E= E(Cord,:);
            end
                 [ ~, label ] = labelFinder( x_InitTrain, y_InitTrain );
                 Cord=find(label(:,2)==1);
                 x_InitTrain= x_InitTrain(Cord,:);
                 y_InitTrain= y_InitTrain(Cord);
                 
        end
        Gk=x_InitTrain;
        APDkn=y_InitTrain;
        APDtrain=APDkn; %%% these will be grown intelligently
        Gktrain=Gk; %%% same for these
        Telapsed=zeros(500,1);

                while length(Gk)<gpoptions.STOP%%%sum(Telapsed)<STOPTIMER%%% Now iterate until you accumulate desired no. of training pnts
                    gpoptions.Batch=gpoptions.Batch+1;
                    tic;
                    [Xnew,APDnew,Enew] = activeLearningPSO(E, Gk, APDkn, gpoptions );
                    Telapsed(time)=toc;
%                     if strcmp(LM,'classifier')
%                     dlmwrite(strcat('Class-',num2str(gpoptions.Batch),'timeTaken.txt'),Telapsed(time));
%                     elseif strcmp(LM,'surface')
%                     dlmwrite(strcat('Surf-',num2str(gpoptions.Batch),'timeTaken.txt'),Telapsed(time));
%                     end
                    time=time+1;
                    if strcmp(LM,'classifier')
                    APDkn=[APDkn;APDnew]; % this is when we use all points grown so far
                    Gk=[Gk;Xnew];
                    tu=randperm(length(APDkn));
                    Gk=Gk(tu,:);
                    APDkn=APDkn(tu); %% Shuffle them
                    APDtrain=[APDtrain;APDnew]; % this is when we use all points grown so far
                    Gktrain=[Gktrain;Xnew];
                    elseif strcmp(LM,'surface')
                    APDkn=[APDkn;APDnew]; % this is when we use all points grown so far
%                     tu=randperm(length(APDkn));
                    Gk=[Gk;Xnew];
%                     Gk=Gk(tu,:);
                    Gktrain=[Gktrain;Xnew];
                    E=Enew;
                    end
                    counter=counter+gpoptions.AScale;
                    disp('loop')
                end
                gpoptions.ALtime=Telapsed;     
                    if strcmp(LM,'classifier')
                         x_train=Gktrain; 
                         y_train=APDtrain;
                    elseif strcmp(LM,'surface') 
                         x_train=Gktrain;
                         HyperParams=gpoptions.classHyperParams;
                         lab=justClassifier(x_InitTrain, y_InitTrain, x_train, HyperParams ) ; 
                         Cord=find(lab==1);
                         x_train= x_train(Cord,:);
                            if size(x_train,2)==2
                                    dummy=ones(length(x_train),2);
                                    x_train2d=cat(2,x_train,dummy);
                                    y_train=EvaluateAPD(x_train2d,gpoptions.pacing); 
                            else%%% option for toy
                                    y_train=EvaluateAPD(x_train,gpoptions.pacing);
                            end
                           
                    end

end

