
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This script runs the iterative active learning for
    % accumulating training points for the O'hara emulator using
    % Gaussian Processes.
    % You can control the following variables to get different
    % behaviour::
    % 1) 'CScale'--No. of initial random sampled data
    % 1) 'AScale'--Active swarm/particle size. For surface fix this to 1
    % 1) 'Rounds**'--No. active learning rounds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
startup
TestData=load('Alearning_2D_10k_Grid.mat');% Change this with 'Alearning_4D_100k_Test.mat' for 4D
CScale=50; %% Initial random data size
AScale=25; %% Active learning swarm size
RoundsSurf=100;
RoundsClass=8;
Telapsed=zeros(1000,1);
STOPCLASS=CScale + RoundsClass*AScale;
STOPSURF=CScale + RoundsSurf*AScale/AScale;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pass al GP related information using the gpoptions structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gpoptions.pacing=100;
gpoptions.NumInducingClass=300; % Inducing points for classifier
gpoptions.sparseMargin=11000; % No of training points upto which we use exact surface inference
gpoptions.NumInducingSurf=1000; % Inducing points for surface
gpoptions.sparseMarginSurf=20000; % No of training points upto which we use exact classifier inference
gpoptions.classHyperParams.minimize=0;
gpoptions.surfHyperParams.minimize=0;
gpoptions.covarianceKernels=@covRQiso;%{'covMaterniso',5};
gpoptions.covarianceKernelsParams=[0.1;0.1;1];%[0.1;1.20];
gpoptions.likelihoodParams=0.015;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Sample Initial Random Training Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
if size(TestData.gk,2)==2
    x_InitTrain = unifrnd(0,1,2,CScale)';
    y_InitTrain=EvaluateAPD([x_InitTrain ones(CScale,2)],gpoptions.pacing);
    X_Grid=TestData.gk;
    Y_Grid=TestData.APDtrue;
    [ ~, labelGrid ] = labelFinder( X_Grid, Y_Grid);
    dlmwrite('GridLabels.txt',labelGrid);
else
    x_InitTrain = unifrnd(0,1,4,CScale)';
    y_InitTrain=EvaluateAPD(x_InitTrain,gpoptions.pacing);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Classifier Learning Phase
%   Learn Hyperparameters for Active Classifier Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gpoptions.LearningMode='classifier';
outparam= learnGPhyp( x_InitTrain, y_InitTrain, gpoptions );
gpoptions.classHyperParams=outparam;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Do a few Rounds of Active Classifier Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gpoptions.Batch=0;
gpoptions.method='pso'; % Flag to trigger the swarming types or grid ('grid' 'pso' 'ga')
gpoptions.Model='APD';
gpoptions.STOP=STOPCLASS;
gpoptions.AScale=AScale;
[ X_ALClassTrain, Y_ALClassTrain ] = intelligentTraining( x_InitTrain, y_InitTrain, gpoptions );
close all

X_class=X_ALClassTrain;
Y_class=Y_ALClassTrain;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Update/Re-learn Hyperparameters after Active Classifier Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gpoptions.LearningMode='classifier';
outparam= learnGPhyp( X_class, Y_class, gpoptions );
gpoptions.classHyperParams=outparam;
TclassTrain=toc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Classifier Learning Phase
%   Learn Hyperparameters for Active Surface Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;
[ ~, label ] = labelFinder( x_InitTrain, y_InitTrain );
Cord=find(label(:,2)==1);
X_surfTrain= x_InitTrain(Cord,:);
Y_surfTrain=y_InitTrain(Cord);
gpoptions.LearningMode='surface';
outparam= learnGPhyp( X_surfTrain, Y_surfTrain, gpoptions );
gpoptions.surfHyperParams=outparam;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Do a few Rounds of Active Surface Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gpoptions.Batch=0;
gpoptions.method='grid'; % Flag to trigger the swarming types or grid ('grid' 'pso' 'ga')
gpoptions.Model='APD';
gpoptions.LearningMode='surface';
gpoptions.STOP=STOPSURF;
gpoptions.AScale=1;
gpoptions.classHyperParams.minimize=0;
gpoptions.surfAlClass=1;
gpoptions.pltDisabled=0;
gpoptions.ALtime=ones(100,1);
[ X_ALSurfTrain, Y_ALSurfTrain ] = intelligentTraining( x_InitTrain, y_InitTrain, gpoptions );
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Filter out only AP related data after Active Surface Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ ~, label ] = labelFinder( X_ALSurfTrain, Y_ALSurfTrain );
Cord=find(label(:,2)==1);
X_ALSurfTrain= X_ALSurfTrain(Cord,:);
Y_ALSurfTrain= Y_ALSurfTrain(Cord);
X_surf=[X_surfTrain;X_ALSurfTrain];
Y_surf=[Y_surfTrain;Y_ALSurfTrain];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Update/Re-learn Hyperparameters after Active Surface Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gpoptions.LearningMode='surface';
if size(X_surf,2)==4
gpoptions.covarianceKernelsParams=[1;1;1];% for 4D the initial values are set like this;
end
outparam= learnGPhyp( X_surf, Y_surf, gpoptions );
gpoptions.surfHyperParams=outparam;

TsurfTrain=toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     Error and Prediction Phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
X_test=TestData.gk;
Y_test=TestData.APDtrue;

%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%% Classifier errors %%%%%%%%%%%%%%%%%%%
[Yp, RepG, yRepG ] = build_multi_domains( X_class, Y_class, X_test, Y_test, gpoptions.classHyperParams );

%%%%%%%%%% True Class Labels %%%%%%%%%%%%%%%%%%
for i=1:length(X_test)
    if Y_test(i)==1000
        Yt(i)=1;
    elseif Y_test(i)==0
        Yt(i)=-1;
    else
        Yt(i)=0;
    end
end
%%%%%%%%%% Predicted Class Labels %%%%%%%%%%%%%%%%%%
MisClassP=0;%%%%build confusion
Yt=Yt+1;Yp=Yp+1;
    for i=1:length(X_test)
        if Yt(i)~=Yp(i)
            MisClassP=MisClassP+1;
        end
    end
ClassifierError=MisClassP; 
PercClassifierError=100*(ClassifierError/length(X_test));
TclassPred=toc;
%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%% Surface errors %%%%%%%%%%%%%%%%%%%
tic;
gpoptions.surfHyperParams.minimize=0;
[PredFinal,UnCert]= pred_scatter_sparse( X_surf, Y_surf, RepG, gpoptions.surfHyperParams );
SurfaceError=mean(abs(yRepG'-PredFinal));
TsurfPred=toc;
%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%% Plot the Final results/surface %%%%%%%%%
plot3(RepG(:,1),RepG(:,2),yRepG,'*','MarkerSize',1);
hold on;
plot3(RepG(:,1),RepG(:,2),PredFinal,'g*','MarkerSize',1);
% Clean-up
command='rm -rf *.txt';
system(command);

