function [ mudL1err,mufL1err,MisClassP ] = ClassifierRegressorTest( Gktrain,APDtrain,gk,APDtrue,hyp1,hyp2,hyp3,SScale)

[ x_train,y_train ] = labelFinder( Gktrain,APDtrain);
n1=length(x_train);n2=n1;n3=n1;
[s1, lpNR ] = pred_class_prob( x_train,y_train(:,1),gk ,n1,hyp1);
[ s2,lpD ] = pred_class_prob( x_train,y_train(:,2),gk ,n2,hyp2);
[s3, lpND ] = pred_class_prob( x_train,y_train(:,3),gk,n3,hyp3);

%%%%%%%%%%%%% Find the Test Data in Depolarizing region %%%%%%%%%%%%%%%%%%%
[RepG,yNoRepG,yRepG,NoDepG,yNoDepG,NoRepG,Yp ] = build_multi_domains( s1,s2,s3,lpNR,lpD,lpND,gk,APDtrue );

%%%%%%%%%% True Class Labels %%%%%%%%%%%%%%%%%%
for i=1:length(gk)
    if APDtrue(i)==1500
        Yt(i)=1;
    elseif APDtrue(i)==10
        Yt(i)=-1;
    else
        Yt(i)=0;
    end
end
MisClassP=0;
    for i=1:length(gk)
        if Yt(i)~=Yp(i)
            MisClassP=MisClassP+1;
        end
    end

%%%%%%%%%%%%% Get a proportion of good data for surface training purpose %%%%%%%%%%%%%%%%%%%
n=length(RepG);
nuu = fix(SScale); tu = randperm(n); tu = tu(1:nuu); 
X_train = RepG(tu,:);
Y_train=yRepG';
Y_train=Y_train(tu,:);
%%%%%%%%%%%%%%%%  pick test points using active learning (differential entropy)    %%%%%%%%%%%%%%%%%%%%%%%%%%
% [~,PredVar]= pred_scatter_sparse(X_train,Y_train,RepG );

%%%%%%%%%%%%%%%%  Predict the surface at test points    %%%%%%%%%%%%%%%%%%%%%%%%%%
[PredFinal,~]= pred_scatter_sparse(X_train,Y_train,RepG );

%%%%%%%%%%%%%%%%  Evaluate Accuracy of predictions  %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%% For Depolarizing region %%%%%%%%%%%%%%%%
depL1err=abs(yRepG'-PredFinal);
%%%%%%% Construct true surface and predict accuracy for true surface %%%%%%%%%%%%%%%%
        j=1;
        k=1;
        l=1;
        for i=1:length(gk)
         [a,b]=max([exp(lpNR(i)),exp(lpD(i)),exp(lpND(i))]);% % 
         [c,d]=min([s1(i),s2(i),s3(i)]);

            if b==1 %&& d==1%%&& APDtrue(i)==1500
                APDpred(i)=yNoRepG(j);
                j=j+1;
            elseif b==2 && APDtrue(i)~=10 && APDtrue(i)~=1500 
                APDpred(i)=PredFinal(k);
                k=k+1;
            elseif b==3 %&& d==3%%&& APDtrue(i)==10
                APDpred(i)=yNoDepG(l);
                l=l+1;
            end

        end

fullL1err=abs(APDtrue-APDpred');
mudL1err=mean(depL1err);
mufL1err=mean(fullL1err);
end

