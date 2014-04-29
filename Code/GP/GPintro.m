%% Setup the GP problem


path(path,'../');

clear all;
close all;

plotfunction = @makePDF;
% plotfunction = @fakePDF; % This one doesn't make output



x = rand(10,1);
noise_ss = 0.05;

test_x = [0:0.01:1]';
test_y = test_x.^3 - 4*test_x.^2 + 2*test_x;

y = x.^3 - 4*x.^2 + 2*x;
y = y + randn(size(x))*sqrt(noise_ss);

plot(x,y,'ko','markersize',20,'linewidth',2);
hold on
plot(test_x,test_y,'b--','linewidth',2)

setupPlot

xlabel('x');
ylabel('y');

yl = ylim;


feval(plotfunction,'gpintro_data.eps');

%% Make some polynomial fits
close all
figure;
plot(x,y,'ko','markersize',20,'linewidth',2);
hold all

X = [x.^0];
testX = [test_x.^0];
plotat = [1 2 3 5];
for order = [1:9]
    X = [X x.^order];
    w = inv(X'*X)*X'*y;
    testX = [testX test_x.^order];
    if any(plotat == order)
        plot(test_x,testX*w,'linewidth',2);
    end
end

legend('Data','1st order','2nd order','3rd order','5th order');

setupPlot
xlabel('x');
ylabel('y');
ylim(yl);
feval(plotfunction,'gpintro_poly.eps');


%% A visual example
close all
kpar = [20 1];
K = kernel(x,x,'gauss',kpar);
testK = kernel(x,test_x,'gauss',kpar);
testKK = kernel(test_x,test_x,'gauss',kpar) + 1e-6*eye(length(test_x));
% Sample functions from the prior
priorFunctions = gausssamp(repmat(0,length(test_x),1),testKK,20);
plot(test_x,priorFunctions','b','linewidth',2)
hold on
plot(x,y,'ko','markersize',20,'linewidth',2);
setupPlot
xlabel('x');
ylabel('y');
ylim(yl);
feval(plotfunction,'gpintro_prior.eps');

close all

% And the posterior
testCov = testKK - testK'*inv(K + noise_ss*eye(length(x)))*testK;
testMu = testK'*inv(K + noise_ss*eye(length(x)))*y;
posteriorFunctions = gausssamp(testMu,testCov,20);

figure;
plot(test_x,posteriorFunctions','b','linewidth',2);
hold on
plot(x,y,'ko','markersize',20,'linewidth',2);
ylim(yl);
setupPlot;
xlabel('x');
ylabel('y');
plot(test_x,testMu,'r','linewidth',2);
feval(plotfunction,'gpintro_posterior.pdf');

% Predictive patch plot
close all
figure
predVar = sqrt(diag(testCov));
patch([test_x' test_x(end:-1:1)'],[testMu'+predVar', testMu(end:-1:1)'-predVar(end:-1:1)'],[0.6 0.6 0.6]);
hold on
plot(test_x,testMu,'r','linewidth',2);
plot(x,y,'ko','markersize',20,'linewidth',2);
setupPlot;
xlabel('x');
ylabel('y');
feval(plotfunction,'gpintro_predictions.pdf');

%% Compute the posterior using Bayes Rule
% Don't actuall use this one...
postCov = inv((1/noise_ss)*eye(length(x)) + K);
postMu = (1/noise_ss)*postCov*y;
close all
postSamples = gausssamp(postMu,postCov,20);
plot(x,postSamples','bo');
hold on
plot(x,y,'ko','markersize',20,'linewidth',2);
xlabel('x');
ylabel('y');
setupPlot;
feval(plotfunction,'gpposterior_bayes.pdf');
%% Hyperparameter plot
close all;
hypvals = [1 10 100];
for i = 1:length(hypvals)
    close all;
    figure
    testKK = kernel(test_x,test_x,'gauss',hypvals(i)) + 1e-6*eye(length(test_x));
    priorFunctions = gausssamp(repmat(0,size(test_x)),testKK,20);
    plot(test_x,priorFunctions,'b','linewidth',2);
    setupPlot;
    xlabel('x');
    ylabel('y');
    filename = sprintf('gpintro_prior_hyp%g.eps',hypvals(i));
    feval(plotfunction,filename);
end

%% Noise-free example
close all
K = kernel(x,x,'gauss',10);
testKK = kernel(test_x,test_x,'gauss',10);
testK = kernel(x,test_x,'gauss',10);

figure
plot(x,y,'ko','markersize',20,'linewidth',2);
testMu = testK'*inv(K + 1e-6*eye(length(x)))*y;
hold on
plot(test_x,testMu,'ro');
for i = 1:length(test_x)
    plot([test_x(i) test_x(i)],[testMu(i) - sqrt(testVar(i)),testMu(i)+sqrt(testVar(i))],'k','color',[0.3 0.3 0.3]);
end

xlabel('x');
ylabel('y');
setupPlot;
feval(plotfunction,'gpintro_noisefree.eps');

%% Noisey example
close all
K = kernel(x,x,'gauss',10);
testKK = kernel(test_x,test_x,'gauss',10);
testK = kernel(x,test_x,'gauss',10);

figure
plot(x,y,'ko','markersize',20,'linewidth',2);
testMu = testK'*inv(K + noise_ss*eye(length(x)))*y;
testVar = diag(testKK) - diag(testK'*inv(K + noise_ss*eye(length(x)))*testK);
hold on
plot(test_x,testMu,'ro');
for i = 1:length(test_x)
    plot([test_x(i) test_x(i)],[testMu(i) - sqrt(testVar(i)),testMu(i)+sqrt(testVar(i))],'k','color',[0.6 0.6 0.6]);
end
xlabel('x');
ylabel('y');
setupPlot;
feval(plotfunction,'gpintro_withnoise.eps');
