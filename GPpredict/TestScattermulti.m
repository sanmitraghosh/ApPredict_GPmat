function [Yt,Yp,APDpred,lpNR,lpD,lpND,s1,s2,s3]=TestScattermulti(Scale,gk,APDtrue,FULL_REGRESS,ALflag,aGk,aAPDkn,hyp1,hyp2,hyp3)
%%%% This is the main function for building GP classifier and regressor

Alearn=0; %% Flag
tic;
CScale=Scale(1);
TScale=Scale(2);
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

%subsample true points to build classifier%
if ALflag==0
nu = fix(CScale); cu = randperm(length(gk)); cu = cu(1:nu); 
Gk = gk(cu,:);
APDkn=APDtrue(cu,:);
APDnew=aAPDkn;
xnew=aGk;
else
Gk=aGk;
APDkn=aAPDkn;
end
%%%%%%%%%%%%%%  ALEARN  %%%%%%%%%%%%%%%%%%%%%%
    if Alearn==1
    APDkn=[APDkn;APDnew];
    Gk=[Gk;xnew];
    tu=randperm(length(APDkn));
    Gk=Gk(tu,:);
    APDkn=APDkn(tu);
    end
%             tu=randperm(length(APDkn));
%             Gk=Gk(tu,:);
%             APDkn=APDkn(tu); %% Shuffle them
%%%% Build NoRep Classifier and predict NoRep class probabilities %%%%%%%
clear y1
x1=Gk;
for i=1:length(x1)
    if APDkn(i)==1500
        y1(i)=1;
    else
        y1(i)=-1;
    end
end
y1=y1';
n1=length(x1);
[s1, lpNR ] = pred_class_prob( x1,y1,gk ,n1,hyp1);

%%%% Build Rep Classifier and predict Rep class probabilities %%%%%%%
clear y2
x2=Gk;
for i=1:length(x2)
    if APDkn(i)==10 
        y2(i)=-1;
    elseif APDkn(i)==1500
        y2(i)=-1;
    else
        y2(i)=1;
    end
end
y2=y2';
n2=length(x2);
[ s2,lpD ] = pred_class_prob( x2,y2,gk ,n2,hyp2);

%%%% Build NoDep Classifier and predict NoDep class probabilities %%%%%%%pred_
clear y3
x3=Gk;
for i=1:length(x3)
    if APDkn(i)==10
        y3(i)=1;
    else
        y3(i)=-1;
    end
end
y3=y3';
n3=length(x3);
tic
[s3, lpND ] = pred_class_prob( x3,y3,gk,n3,hyp3);
TLC=toc;
%%%% One Versus All class predictions and build domains %%%%%%%
[RepG,yNoRepG,yRepG,NoDepG,yNoDepG,NoRepG,Yp ] = build_multi_domains( s1,s2,s3,lpNR,lpD,lpND,gk,APDtrue );

if FULL_REGRESS==1
        [PredFinalRepG,hypRegress] = pred_scatter_sparse( RepG,yRepG',TScale );

        %%%% Prediction for Entire Surface %%%%%%%

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
                APDpred(i)=PredFinalRepG(k);
                k=k+1;
            elseif b==3 %&& d==3%%&& APDtrue(i)==10
                APDpred(i)=yNoDepG(l);
                l=l+1;
            end

        end

end



end
