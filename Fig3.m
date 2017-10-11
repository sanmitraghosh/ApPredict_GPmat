

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This script gathers actively learnt training data for
    % classifier/boundary prediction. This corresponds to the Fig3 in the paper.
    % You can control the following variables to get different
    % behaviour::
    % 1) 'CScale'--No. of initial random sampled data
    % 1) 'AScale'--Active swarm/particle size. For surface fix this to 1
    % 1) 'Rounds'--No. active learning rounds
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
startup
load('Alearning_4D_100k_Train.mat');%Change this with 'Alearning_4D_100k_Train.mat' for 4D

GridData=load('Alearning_2D_10k_Grid.mat');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pass al GP related information using the gpoptions structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

CScale=500;
AScale=1;
Rounds=2500;
STOPSURF=CScale + Rounds*AScale;
gpoptions.pacing=100;
gpoptions.NumInducingClass=300;
gpoptions.sparseMargin=5000;
gpoptions.NumInducingSurf=1000;
gpoptions.sparseMarginSurf=10000;
gpoptions.classHyperParams.minimize=0;
gpoptions.surfHyperParams.minimize=0;
gpoptions.covarianceKernels=@covNNone;%@covRQiso;%{'covMaterniso',5};
gpoptions.covarianceKernelsParams=[0.1;1];%[0.1;0.21;1];%[0.1;1.20];
gpoptions.likelihoodParams=0.015;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load up a pool of random training data
    % from which we will draw randomely to compare with active learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_random=gk;
Y_random=APDtrue;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Use the gridded data for visualisations for the 2D case
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if size(gk,2)==2
    X_Grid=GridData.gk;
    Y_Grid=GridData.APDtrue;
    [ ~, labelGrid ] = labelFinder( X_Grid, Y_Grid);
    dlmwrite('GridLabels.txt',labelGrid);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Sample Initial Random Training Data from the pool loaded earlier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nu = fix(CScale); cu = randperm(length(X_random)); cu = cu(1:nu); 
x_Initrandom = X_random(cu,:);
y_Initrandom = Y_random(cu,:); %%% Start with random CScale points
X_random(cu,:)=[]; %%% Remove these chosen points from the pool
Y_random(cu)=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Learn a classifier to constrain the chosen active points in the AP
%   region
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

gpoptions.LearningMode='classifier';
outparam= learnGPhyp( x_Initrandom, y_Initrandom, gpoptions );
gpoptions.classHyperParams=outparam;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Learn hyperparameters for surface active learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[ ~, label ] = labelFinder( x_Initrandom, y_Initrandom );
Cord=find(label(:,2)==1);
x_InitTrain= x_Initrandom(Cord,:);
y_InitTrain= y_Initrandom(Cord);
gpoptions.LearningMode='surface';
outparam= learnGPhyp( x_InitTrain, y_InitTrain, gpoptions );
gpoptions.surfHyperParams=outparam;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Do a few rounds of surface active learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gpoptions.Batch=0;
gpoptions.method='grid'; % Flag to trigger the swarming types or grid ('grid' 'pso' 'ga')
gpoptions.Model='APD';
gpoptions.LearningMode='surface';
gpoptions.STOP=STOPSURF;
gpoptions.AScale=AScale;
gpoptions.classHyperParams.minimize=0;
gpoptions.surfAlClass=1;
gpoptions.pltDisabled=1; % Turning it on starts plotting contour plots (slow)
gpoptions.ALtime=ones(100,1);
[ X_intellitrain, Y_intellitrain ] = intelligentTraining( x_Initrandom, y_Initrandom, gpoptions );
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Keep only the AP related points for the active training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ ~, label ] = labelFinder( X_intellitrain, Y_intellitrain );
Cord=find(label(:,2)==1);
x_train= X_intellitrain(Cord,:);
y_train= Y_intellitrain(Cord);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Keep only the AP related points for random training data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ ~, label ] = labelFinder( X_random, Y_random );
Cord=find(label(:,2)==1);
X_train= X_random(Cord,:);
Y_train= Y_random(Cord);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Save the dataset, please rename accordingly!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save('surfActive2D.mat')
% Clean-up
command='rm -rf *.txt';
system(command);
if size(gk,2)==2
    dlmwrite('GridLabels.txt',labelGrid);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Call up evaluation script to compare with random learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Fig3a
