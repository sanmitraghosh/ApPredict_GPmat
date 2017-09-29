function [APD]= EvaluateAPD_old(gk,Init_Pace)
%%%% Solve O'hara model and calculate APDs %%%%
dim=size(gk,2);
LastState=ones(42,1);
pa=[1 1 1 1];

    for k=1:Init_Pace
        if k==1
        [Yfirst, ~] = ohara(pa,LastState,1);
        else
        [Yfirst, ~] = ohara(pa,Yfirst(end,:),2);  
        end
    end
    
    SavedStates=Yfirst(end,:);
%%%%%%%%%%%%%%%%%%%  Find Threshold by blocking Na %%%%
   for j=1:Init_Pace-1
        if j==1
        [YNoRep ~] = ohara(pa,SavedStates,1);
        else
        [YNoRep time] = ohara(pa,YNoRep(end,:),2);  
        end
   end
[Tcurve ~] = ohara([0 1 1 1],YNoRep(end,:),2);
THRNODEP=min(Tcurve(:,1)) + 1.1*(max(Tcurve(:,1)) - min(Tcurve(:,1)) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%           
% errcnt=1;
parfor i=1:size(gk,1)

    Pa=[gk(i,1) gk(i,2) gk(i,3) gk(i,4)];
    disp(i)
   for j=1:Init_Pace-1
        if j==1
        [Yinter ~] = ohara(Pa,SavedStates,2);
        else
        [Yinter time] = ohara(Pa,Yinter(end,:),2);  
        end
   end
[Y Time] = ohara(Pa,Yinter(end,:),2);
%[APD(i)]=FindAPD(Y(:,1),Time,THRNODEP);
try
    [APD(i)]=FindAPD(Y(:,1),Time,THRNODEP);
catch
    warning(strcat('fuzzy APD value.',num2str(i)));
    APD(i) = -10;
    disp(strcat('problem  ',num2str(i)));
%     errcnt=errcnt+1;
end

end
APD=APD'
% APDchaste=FindAPD(chaste(:,2),Time)







% clear all
% 
% % 
% Pa=gk(2690,:)
% % 
% [Y Time] = ohara(Pa,Yinter(end,:),2);
% [APD,Tp(i),Tr(i)]=FindAPD(Y(:,1),Time);
% [APD,Tp,Tr]=FindAPD(Y(:,1),Time);
