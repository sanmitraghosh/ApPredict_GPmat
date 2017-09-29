close all
clear all

d = importdata('/home/sanosh/eclipse/workspace/chaste/testoutput/InterpolationError/InterpolationError.dat');
data = d.data;
surf=load('Fig1lsurfCurve.mat');
class=load('Fig1lclassCurve.mat');
num_test_points = 100000;

for i=1:size(data,1)
    assert(sum(data(i,3:end))==num_test_points); % Check nothing funny is going on!
    misclassification_rate(i) = (num_test_points - data(i,3) -data(i,7) - data(i,11))./num_test_points;
end


figure
subplot(2,2,1)
plot(data(:,1),data(:,2),'r-','LineWidth',1.5)
hold on
plot(surf.CScale,surf.Error,'b--','LineWidth',1.5)
legend('Interpolation','GP')
ylabel('L1 Error in APD90 (ms)')
xlabel('Number of Training Points')
xlim([min(data(:,1)) max(data(:,1))])

subplot(2,2,2)
loglog(data(:,1),data(:,2),'r-','LineWidth',1.5)
hold on
loglog(surf.CScale,surf.Error,'b--','LineWidth',1.5)
ylabel('L1 Error in APD90 (ms)')
xlabel('Number of Training Points')
xlim([min(data(:,1)) max(data(:,1))])

subplot(2,2,3)
plot(data(:,1),100.0.*misclassification_rate,'b-','LineWidth',1.5)
hold on
plot(class.CScale,100*(class.Error/num_test_points),'g--','LineWidth',1.5)
legend('Interpolation','GP')
ylabel('Misclassification Rate (%)')
xlabel('Number of Training Points')
xlim([min(data(:,1)) max(data(:,1))])

subplot(2,2,4)
loglog(data(:,1),100.0.*misclassification_rate,'b-','LineWidth',1.5)
hold on
loglog(class.CScale,100*(class.Error/num_test_points),'g--','LineWidth',1.5)
ylabel('Misclassification Rate (%)')
xlabel('Number of Training Points')
xlim([min(data(:,1)) max(data(:,1))])
