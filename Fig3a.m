
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % This script compares the active learning with random learning for
    % surface prediction. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
clear all
startup
ActiveData=load('surfActive2D.mat');
TestData=load('Alearning_2D_10k_Grid.mat'); % Change this with 'Alearning_4D_100k_Test.mat' for 4D
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Pass al GP related information using the gpoptions structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
gpoptions.NumInducingSurf=1000;
gpoptions.sparseMarginSurf=10000;
gpoptions.classHyperParams.minimize=0;
gpoptions.surfHyperParams.minimize=0;
gpoptions.covarianceKernels={'covMaterniso',5};
gpoptions.covarianceKernelsParams=[0.1;1.20];
gpoptions.likelihoodParams=0.015;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load up the initial, random training, and test dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

X_init=ActiveData.x_InitTrain; %initial training data
Y_init=ActiveData.y_InitTrain;


X_train=ActiveData.X_train; % random training data
Y_train=ActiveData.Y_train;

X_test=TestData.gk; % test dataset
Y_test=TestData.APDtrue;
[ ~, label ] = labelFinder( X_test, Y_test );
Cord=find(label(:,2)==1);
X_test= X_test(Cord,:);
Y_test= Y_test(Cord);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test surface error on random data (draw 10 times to get average)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
CScale=[length(X_init):10:length(ActiveData.x_train)];
[t1,t2] = ndgrid(linspace(0,1,100)); 
grid2D = [t1(:),t2(:)];
for k=1:10
        nu = fix(length(ActiveData.x_train)); cu = randperm(length(X_train)); cu = cu(1:nu); 
        x_train = X_train(cu,:);
        y_train = Y_train(cu);
    for j=1:length(CScale)
        if j==1
            x_dumb=X_init;
            y_dumb=Y_init;  
            X_active=X_init;%%% For viz contour
            Y_active=Y_init;
        else
            x_dumb=[X_init; x_train(1:CScale(j)-CScale(1),:)];
            y_dumb=[Y_init; y_train(1:CScale(j)-CScale(1))]; 
            X_active=ActiveData.x_train(1:CScale(j),:);
            Y_active=ActiveData.y_train(1:CScale(j)); 
        
        end
%%%%%%%%%%%%%%%%%%%% This bit is for visualisation in the 2D case %%%%%%%%%
       if k==1
            if size(x_dumb,2)==2
                contMapD=surfaceCertainty( x_dumb, y_dumb, grid2D, ActiveData.outparam );
                contMapA=surfaceCertainty( X_active, Y_active, grid2D, ActiveData.outparam );
                swarm=[X_active x_dumb];
                plotDumbContours( swarm,j,contMapA, contMapD, 'surface' )
            end
       end 
        gpoptions.surfHyperParams=ActiveData.outparam;
        gpoptions.surfHyperParams.minimize=0;
        [PredFinalDumb,UnCertDumb]= pred_scatter_sparse( x_dumb, y_dumb, X_test, gpoptions.surfHyperParams );
        SurfaceErrorDumb(j,k)=mean(abs(Y_test-PredFinalDumb));
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Test surface error on active learning data 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j=1:length(CScale)
        if j==1
        X_active=X_init;
        Y_active=Y_init;
        else
        X_active=ActiveData.x_train(1:CScale(j),:);
        Y_active=ActiveData.y_train(1:CScale(j)); 
        
        end
        gpoptions.surfHyperParams=ActiveData.outparam;
        gpoptions.surfHyperParams.minimize=0;
        
        [PredFinalActive,UnCertActive]= pred_scatter_sparse( X_active, Y_active, X_test, gpoptions.surfHyperParams );
        SurfaceErrorActive(j)=mean(abs(Y_test-PredFinalActive));

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot Error Curves
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all
errorbar(CScale,mean(SurfaceErrorDumb,2),std(SurfaceErrorDumb'),'LineWidth',3)
hold on
plot(CScale,SurfaceErrorActive,'LineWidth',3)
xlim([CScale(1) CScale(end)])
xlabel('Training Size')
ylabel('mean absolute error')
legend('Dumb Error (mean out of 10 runs)', 'Active Error')
title('Surface Active Learning')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot predicted Surface
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure 
plot3(X_test(:,1),X_test(:,2),PredFinalActive,'*','MarkerSize',1)
hold on
plot3(X_test(:,1),X_test(:,2),Y_test,'*','MarkerSize',1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Plot prediction errors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Error=abs(Y_test-PredFinalActive);
LocError=find(Error>20);
length(LocError)
figure
plot3(X_test(:,1),X_test(:,2),Error,'*','MarkerSize',1)

