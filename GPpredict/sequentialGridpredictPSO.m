function [X,APD,Enew] = sequentialGridpredictPSO( E,Gk,APDkn,hyp1, hyp2, hyp3, method, AScale, Init_Pace,gpoptions )
%%%%%% This function is used to iteratively add training points %%%%
 %%% choose 1000 points using certainty measure
x=Gk;
Model=gpoptions.Model;
Batch=gpoptions.Batch;
%%%%% This section is data preparation (class labels) for certainty predictions %%%%
for i=1:length(x)
    if APDkn(i)==1000
        y1(i)=1;
    else
        y1(i)=-1;
    end
end
y1=y1';

for i=1:length(x)
    if APDkn(i)==0 
        y2(i)=-1;
    elseif APDkn(i)==1000
        y2(i)=-1;
    else
        y2(i)=1;
    end
end
y2=y2';

for i=1:length(x)
    if APDkn(i)==0
        y3(i)=1;
    else
        y3(i)=-1;
    end
end
y3=y3';
y=[y1 y2 y3];
%%%%%%%%%%%% Create inducing points for using sparse covariances %%%%%%%
                dim=size(Gk,2);
                Ind=300;
                sparse=nthroot(Ind,dim);
               switch dim
                   case 4
                    [u1,u2,u3,u4] = ndgrid(linspace(0,1,sparse)); 
                    u = [u1(:),u2(:),u3(:), u4(:)]; 
                    clear u1; clear u2;clear u3;
                   case 3
                    [u1,u2,u3] = ndgrid(linspace(0,1,sparse)); 
                    u = [u1(:),u2(:),u3(:)]; 
                    clear u1; clear u2;clear u3;
                   case 2
                    [u1,u2] = ndgrid(linspace(0,1,sparse)); 
                    u = [u1(:),u2(:)]; 
                    clear u1; clear u2;
                end
%                 nu = size(u,1);    
%%%%%%%%%%%% Prepare GP %%%%%%%
            meanfunc = @meanConst; 
            covfunc = @covSEard;  
            covfuncF = {@covFITC, {covfunc},u};
            likfunc = @likErf;
%%%%%%%%%%%% Setup the optimisation problem, or bypass swarming for gridding %%%%%%%
            if strcmp(Model,'Toy')
                if length(x)<1000
                        ObjectiveFunction=@(swarm) certainty( x, y, swarm, meanfunc, covfunc,  likfunc, hyp1, hyp2, hyp3 );
                else
                        ObjectiveFunction=@(swarm) certainty( x, y, swarm, meanfunc, covfuncF,  likfunc, hyp1, hyp2, hyp3 );
                end
            elseif strcmp(Model,'APD')
                if length(x)<1000
                        ObjectiveFunction=@(swarm) certainty( x, y, swarm, meanfunc, covfunc, likfunc, hyp1, hyp2, hyp3 );
                else
                        ObjectiveFunction=@(swarm) certainty( x, y, swarm, meanfunc, covfuncF, likfunc, hyp1, hyp2, hyp3 );
                end
            end
            nvars=dim;
            LB=zeros(1,dim);
            UB=ones(1,dim);
            
%%%%%%%%%%%% Run PSO or GA for point selection %%%%%%%
switch method
    case 'ga' 
            [Xinit1,Xinit2] = ndgrid(linspace(0,1,25));
            Xinit=[Xinit1(:),Xinit2(:)]; %% create grid
            P=certainty( x, y, Xinit, meanfunc, covfuncF, likfunc, hyp1, hyp2, hyp3 );
            [~,Cert]=sort(P);
            Xinit=Xinit(Cert(1:AScale),:); %%% This bit is to initialize the popultn using certainty on some random pnts %%
            optionsga = gaoptimset('PopulationSize',AScale,'Vectorized','on','Display','iter','PopInitRange',[-100;100],...,
            'OutputFcn',@gaoutfun,'CrossoverFraction',0.52,'TimeLimit',1200)%'SelectionFcn',@selectiontournament,
            [xbest,fval,~,~,X] = ga(ObjectiveFunction,nvars,[],[],[],[],LB,UB,[],optionsga)
            APD=EvaluateAPD(X,Init_Pace); %%%Uncomment this when swarming
            Enew=E;
    case 'pso'
%%%%%%%%%%%% This commented bit is for 2D vdo generation only %%%%%%%
%             dlmwrite('psoBatch.txt',Batch);%%%for video
%             [t1,t2] = ndgrid(linspace(0,1,100)); 
%             t = [t1(:),t2(:)]; 
%             if length(x)<1000
%             contMap=certainty( x, y, t, meanfunc, covfunc, likfunc, hyp1, hyp2, hyp3 );  
%             else
%             contMap=certainty( x, y, t, meanfunc, covfuncF, likfunc, hyp1, hyp2, hyp3 );
%             end
%             dlmwrite('contMap.txt',contMap);
            if strcmp(Model,'APD')
            Xinit=unifrnd(0.1,0.9,AScale,dim);
%             P=certainty( x, y, Xinit, meanfunc, covfuncF, likfunc, hyp1, hyp2, hyp3 );
            elseif strcmp(Model,'Toy') %%%% only used to train toy surface
            Xinit=unifrnd(0.1,0.9,AScale,dim);
%             [Xinit1,Xinit2,Xinit3,Xinit4] = ndgrid(linspace(0,1,25));
%             Xinit=[Xinit1(:),Xinit2(:),Xinit3(:),Xinit4(:)]; %% create grid
%             P=certainty( x, y, Xinit, meanfunc, covfunc, likfunc, hyp1, hyp2, hyp3 );
            end
%             [~,Cert]=sort(P);
%             Xinit=Xinit(Cert(1:AScale),:); %%% This bit is to initialize the popultn using certainty on some random pnts %%
            optionspso = optimoptions(@particleswarm,'OutputFcn',@pswplotranges,'MaxIter',130,...,
                'InitialSwarmSpan',0.5,'InitialSwarm',Xinit,'Vectorized','on','UseParallel',true,'Display','iter','SwarmSize',AScale,'MinFractionNeighbors',0.25)
            optionspso.TolFun=1e-3;
%             optionspso.UseParallel=1;
%             optionspso.InitialSwarm=[];
            [xbest,fval,exitflag,output] = particleswarm(ObjectiveFunction,nvars,LB,UB,optionspso)%%% can't get the population as output
            X = dlmread('myFile.txt')
            if strcmp(Model,'APD')
            APD=EvaluateAPD(X,Init_Pace); %%%Uncomment this when swarming
            elseif strcmp(Model,'Toy')
            for i=1:length(X)
                APD(i)=surfaceGenerate(X(i,1),X(i,2));
            end
            APD=APD';
            end
            Enew=X;
    case 'grid' 
            if strcmp(Model,'Toy')
            P=certainty( x, y, E, meanfunc, covfunc, likfunc, hyp1, hyp2, hyp3 );
            elseif strcmp(Model,'APD')
            P=certainty( x, y, E, meanfunc, covfuncF, likfunc, hyp1, hyp2, hyp3 );
            end
            [~,Cert]=sort(P);
            X=E(Cert(1:AScale),:);
            if strcmp(Model,'APD')
            APD=EvaluateAPD(X,Init_Pace); %%%Uncomment this when swarming
            elseif strcmp(Model,'Toy')
            for i=1:length(X)
                APD(i)=surfaceGenerate(X(i,1),X(i,2));
            end
            APD=APD';
            end

            E(Cert(1:AScale),:)=[];
            Enew=E; %%% remove good points from the grid for next iteration
 end

end

