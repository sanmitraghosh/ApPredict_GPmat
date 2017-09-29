 
function [APD,Tp,Tr]= FindAPD(curve,t,THRNODEP)

%%% Calculate APD for a single action Potential.
trough=zeros(10,1);  
peak=zeros(10,1);
THR=min(curve) + 0.1*(max(curve) - min(curve) );
j=0;
   for i=1:length(curve)
       if curve(i)>THR%curve(i-1)  && curve(i)>0
           j=j+1;
                  
           if j==1
                 peak2=curve(i);
                 peak1=curve(i-1);
                 TP2=t(i);
                 TP1=t(i-1);
%                  peak=i
                 index=i;
                 break;
           end
       end
   end
   k=0;
   NoRep=0;%%%A Flag that points out Repolarization occured or not. '0' for normal repolarizing%%%
   for i=index:length(curve)
       if curve(i)<THR%0.1*curve(peak)&& curve(i)>0
           k=k+1;
           if k==1
                 trough2=curve(i);
                 trough1=curve(i-1);
                 TR2=t(i);
                 TR1=t(i-1);
                 NoRep=0;
             break;
           end
       else
        NoRep=1;
       end
   end 
 
 if NoRep==1
     Tr=length(curve);
 else
Tr=(((trough1-(THR))/(trough1-trough2))*(TR2-TR1))+TR1; %%% this interpolation gives the axact boundaries
 end
Tp=(((THR-peak1)/(peak2-peak1))*(TP2-TP1))+TP1;
   APD=Tr-Tp;
   if APD>900
       APD=1000;
   elseif max(curve)<THRNODEP%-50
       APD=0; 
%    elseif curve(15+10)<0 && max(curve)>0
%        APD=10;
   end
  end

           
