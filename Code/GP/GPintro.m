%% Setup the GP problem

path(path,'../');

clear all;
close all;

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


makePDF('gpintro_data.eps');

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
makePDF('gpintro_poly.eps');


%% A visual example
close all
K = kernel(x,x,'gauss',20);
testK = kernel(x,test_x,'gauss',20);
testKK = kernel(test_x,test_x,'gauss',20) + 1e-6*eye(length(test_x));
% Sample functions from the prior
priorFunctions = gausssamp(repmat(0,length(test_x),1),testKK,20);
plot(test_x,priorFunctions','b','linewidth',2)
hold on
plot(x,y,'ko','markersize',20,'linewidth',2);
setupPlot
xlabel('x');
ylabel('y');
ylim(yl);
makePDF('gpintro_prior.eps');

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
plot(test_x,te,'b','linewidth',stMu2);
makePDF('gpintro_posterior.pdf');