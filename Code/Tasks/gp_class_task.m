%% gp_class_task
clear all;
close all;

% If using Octave, uncomment the following line
% more off

% Classification example
% Generate some data
gam = 0.1; % Covariance parameter, lower = smoother
close all;

% Data from three Gaussians
x = [randn(20,2)+repmat([2 0],20,1);
    randn(20,2) + repmat([-2 0],20,1);
    randn(30,2) + repmat([0 2],30,1)];

y = [repmat(1,40,1);repmat(-1,30,1)];
N = length(y);

% Generate a grid for evaluating test probabilities
xl = [-5 5];
yl = [-3 5];
[X,Y] = meshgrid(xl(1):0.2:xl(2),yl(1):0.2:yl(2));
testx = [reshape(X,prod(size(X)),1) reshape(Y,prod(size(Y)),1)];

% Create the covariance functions
C = kernel(x,x,'gauss',gam) + 1e-6*eye(N);
testC = kernel(testx,x,'gauss',gam);


% Do some sampling
Ci = inv(C); % Only need to compute this once
postC = inv(Ci + eye(N));
f = gausssamp(repmat(0,N,1),C,1)'; % Initialise f
z = zeros(size(f));

NITS = 500; % More samples = smoother contours
BURN = 100; % Ignore the first BURN samples
allf = zeros(NITS,N);

testprobs = zeros(size(testx,1),1);

% Precompute some things that don't change
Cprod = diag(testC*(C\testC'));
testvar = repmat(1,size(testx,1),1) - Cprod;

for it = 1:NITS
    fprintf('Iteration %g\n',it);
    for n = 1:N
        % sample z_n from a truncated Gaussian with mean f_n
        % truncated to be positive if y_n=1 and negative if y_n=-1
        % uses simple rejection sampling
        z(n) = randn + f(n);
        while z(n)*y(n)<0
            z(n) = randn + f(n);
        end
    end
    
    % Sample f
    f = gausssamp(postC*z,postC,1)';

    % Do a GP regression for predictions
    testmu = (testC/C)*f;
    % Generate a sample of test f from the GP
    testsample = randn(size(testx,1),1).*sqrt(testvar) + testmu;
    if it>BURN
        testprobs = testprobs + normcdf(testsample);
        % Alternatively compute it by sampling z and seeing if it is +ve:
%         testz = randn(size(testsample)) + testsample;
%         testprobs = testprobs + (testz>0);
    end
    allf(it,:) = f';
end

% normalise the testprobs
testprobs = testprobs ./ (NITS-BURN);
% reshape them back to grid shape
testprobs = reshape(testprobs,size(X));
% Make the plot
close all
plotClassData(x,y);
xlim(xl);
ylim(yl);
[c,h] = contour(X,Y,testprobs);
clabel(c,h,'color','b','fontsize',14)   


