%% GP task script
clear all; close all;

% generate a dataset
N = 20;
x = sort(rand(N,1));
y = randn(N,1) + 2*x; % Random targets - replace with anything you like
figure
plot(x,y,'ko','markersize',10);


% Define the hyper-parameters - try changing these
gam = 1; % Low gam = smooth functions
ss = 2;

% create the covariance matrix
C = kernel(x,x,'gauss',gam) + 1e-6*eye(N);
imagesc(C);
title('Visualised covariance matrix');


% Compute posterior covariance
postC = inv(inv(C) + (1/ss)*eye(N));
postMu = postC*y;

% Plot the data, posterior mean, and plus/minus 3 standard deviations
figure
plot(x,y,'ko','markersize',10);
hold on
errorbar(x,postMu,3*sqrt(diag(postC)),'.')
title('Data and posterior mean and 3 * sd');
axis tight
% Predictive function
% define some test points
testx = [0:0.01:1]';
testN = length(testx);
testC = kernel(testx,x,'gauss',gam);
predMu = testC*inv(C + ss*eye(N))*y;
predCov = kernel(testx,testx,'gauss',gam) - testC*inv(C + ss*eye(N))*testC';

figure
plot(x,y,'ko','markersize',10)
hold on
errorbar(testx,predMu,3*sqrt(diag(predCov)),'r.')
title('Predicted mean at test points and 3 * sd and some posterior sampels')
% Generate some posterior samples
postSamp = gausssamp(predMu,predCov + 1e-6*eye(testN),20);
plot(testx,postSamp,'k','color',[0.6 0.6 0.6])
axis tight

% Sample some functions from the prior
figure
plot(x,y,'ko','markersize',10)
hold on
bigpriorCov = kernel(testx,testx,'gauss',gam) + 1e-6*eye(testN);
priorSamples = gausssamp(repmat(0,testN,1),bigpriorCov,20);
plot(testx,priorSamples,'k','color',[0.6 0.6 0.6]);
title('Data and samples from the prior');
