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
testVar = diag(testKK) - diag(testK'*inv(K + noise_ss*eye(length(x)))*testK);
ylim(yl);
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

%% Classification example
% Generate some data
gamvals = [1 5 10];
close all;
x = [randn(20,2)+repmat([2 0],20,1);
    randn(20,2) + repmat([-2 0],20,1);
    randn(30,2) + repmat([0 2],30,1)];

y = [repmat(1,40,1);repmat(-1,30,1)];


[X,Y] = meshgrid(xl(1):0.2:xl(2),yl(1):0.2:yl(2));
testx = [reshape(X,prod(size(X)),1) reshape(Y,prod(size(Y)),1)];

for g = 1:length(gamvals)
    gam = gamvals(g);
    testC = kernel(testx,x,'gauss',gam);

    % Do some sampling
    C = kernel(x,x,'gauss',gam);
    Ci = inv(C);
    N = size(x,1);
    f = gausssamp(repmat(0,N,1),C,1)';
    z = zeros(size(f));
    NITS = 500;
    BURN = 100;
    postC = inv(Ci + eye(N));
    allf = zeros(NITS,N);
    testprobs = zeros(size(testx,1),1);
    testvar = zeros(size(testx,1),1);
    Cprod = diag(testC*(C\testC'));
    testvar = repmat(1,size(testx,1),1) - Cprod;
    for it = 1:NITS
        fprintf('Iteration %g\n',it);
        for n = 1:N
            % sample z_n
            z(n) = randn + f(n);
            while z(n)*y(n)<0
                z(n) = randn + f(n);
            end
        end
        % Sample f
        f = gausssamp(postC*z,postC,1)';

        % Do the regression for predictions
        testmu = (testC/C)*f;

        testsample = randn(size(testx,1),1).*sqrt(testvar) + testmu;
        if it>BURN
            testprobs = testprobs + normcdf(testsample);
        end
        allf(it,:) = f';
    end

    testprobs = testprobs ./ (NITS-BURN);
    testprobs = reshape(testprobs,size(X));
    %% Make the plot
    close all
    plotClassData(x,t);
    setupPlot;
    xl = [-5 5];
    yl = [-3 5];
    xlim(xl);
    ylim(yl);
    feval(plotfunction,'class_data.eps');
    [c,h] = contour(X,Y,testprobs,'k','color',[0.3 0.3 0.3]);
    clabel(c,h,'color','b','fontsize',14)   
    fname = sprintf('gpclass_hyp%g.eps',gam);
    feval(plotfunction,fname);
    
    close all
    surf(X,Y,testprobs);
    view([0 0 1]);
    hold on
    plotClassData3(x,t);
    setupPlot;
    fname = sprintf('gpclass_hyp%g_surf.eps',gam);
    feval(@makePDFopengl,fname);
end

%% Generative process
for i = 1:3
    close all
    N = 100;
    x = randn(N,2)*10;
    gam = 0.005;
    C = kernel(x,x,'gauss',gam) + 1e-6*eye(N);
    f = gausssamp(zeros(N,1),C,1)';
    z = randn(N,1).*0.1 + f;
    y = zeros(N,1);
    y(z<0) = 0;
    y(z>0) = 1;
    plotClassData(x,y);
    setupPlot;
    fname = sprintf('gendata%g.eps',i);
    makePDF(fname);
end
%% Multi-class example
x = [randn(20,2)+repmat([2 0],20,1);
    randn(20,2) + repmat([-2 0],20,1);
    randn(30,2) + repmat([0 2],30,1)];

y = [repmat(1,20,1);repmat(2,20,1);repmat(3,30,1)];


[X,Y] = meshgrid(xl(1):0.2:xl(2),yl(1):0.2:yl(2));
testx = [reshape(X,prod(size(X)),1) reshape(Y,prod(size(Y)),1)];

gam = 1;
testC = kernel(testx,x,'gauss',gam);

% Do some sampling
C = kernel(x,x,'gauss',gam);
Ci = inv(C);
N = size(x,1);

K = max(y);
for k = 1:K
    f(:,k) = gausssamp(repmat(0,N,1),C,1)';
end

z = zeros(size(f));
NITS = 5000;
BURN = 1000;
postC = inv(Ci + eye(N));
testprobs = zeros(size(testx,1),K);
testvar = zeros(size(testx,1),1);
Cprod = diag(testC*(C\testC'));
testvar = repmat(1,size(testx,1),1) - Cprod;

for it = 1:NITS
    fprintf('Iteration %g\n',it);
    for n = 1:N
        % sample z_n
        z(n,:) = randn(1,K) + f(n,:);
        pos = find(z(n,:)==max(z(n,:)),1);
        while pos ~= y(n)
            z(n,:) = randn(1,K) + f(n,:);
            pos = find(z(n,:)==max(z(n,:)),1);
        end
    end
    % Sample f
    for k = 1:K
        f(:,k) = gausssamp(postC*z(:,k),postC,1)';
    end
    
    % Do the regression for predictions
    
    testmu = (testC/C)*f;
    

    testsample = randn(size(testx,1),K).*repmat(sqrt(testvar),1,K) + testmu;
    testz = testsample + randn(size(testx,1),K);
    if it > BURN
        for n = 1:size(testx,1)
            pos = find(testz(n,:) == max(testz(n,:)),1);
            testprobs(n,pos) = testprobs(n,pos) + 1;
        end
    end
end

testprobs = testprobs ./ repmat(sum(testprobs,2),1,K);

for k = 1:K
    close all
    surf(X,Y,reshape(testprobs(:,k),size(X)));
    view([0 0 1]);
    hold on
    plotClassData3(x,y);
    setupPlot;
    fname = sprintf('multiclass_k%g.eps',k);
    feval(@makePDFopengl,fname);
end


close all
plotClassData3(x,y);
setupPlot;
fname = 'multiclass_data.eps';
feval(@makePDF,fname);


%% Visualising aux variable trick
close all
f = 0.7;
xvals1 = [-3:0.01:0];
xvals2 = [0:0.01:5];
p1x = [xvals1 xvals1(end) xvals1(1)];
p1y = [normpdf(xvals1,f,1),0,0];

p2x = [xvals2 xvals2(end) xvals2(1)];
p2y = [normpdf(xvals2,f,1),0,0];
patch(p1x,p1y,[0.8 0.8 0.8]);
patch(p2x,p2y,[0.6 0.6 0.6]);
axis tight
hold on
plot([0 0],ylim,'k--');
% text(f+0.1,0.1,'$f_n$','interpreter','latex','fontsize',24)
text(0.5,0.2,'$y_n=1$','interpreter','latex','fontsize',24)
text(-2.5,0.2,'$y_n=-1$','interpreter','latex','fontsize',24)
setupPlot
xlabel('$z_n$','interpreter','latex','fontsize',24)
makePDF('vis_aux.eps');

%% And with the ncnm
close all
f = 0.7;
xvals1 = [-3:0.01:-0.5];
xvals2 = [0.5:0.01:5];
p1x = [xvals1 xvals1(end) xvals1(1)];
p1y = [normpdf(xvals1,f,1),0,0];

p2x = [xvals2 xvals2(end) xvals2(1)];
p2y = [normpdf(xvals2,f,1),0,0];
patch(p1x,p1y,[0.8 0.8 0.8]);
patch(p2x,p2y,[0.6 0.6 0.6]);
axis tight
hold on
plot([-0.5 -0.5],ylim,'k--');
plot([0.5 0.5],ylim,'k--');
% text(f+0.1,0.1,'$f_n$','interpreter','latex','fontsize',24)
text(1.0,0.2,'$y_n=1$','interpreter','latex','fontsize',24)
text(-2.7,0.2,'$y_n=-1$','interpreter','latex','fontsize',24)
h = text(0,0.15,'$y_n=0$','interpreter','latex','fontsize',24,'rotation',90)
setupPlot
xlabel('$z_n$','interpreter','latex','fontsize',24)
makePDF('vis_aux_ncnm.eps');


%% NCNM example
close all
data = twospirals(400,100,5,100.0);
y = data(:,3);x = data(:,1:2);
y(y==0) = -1;
N = size(y,1);
g = zeros(N,1);

% pos = find(y==-1);
% x(pos,2) = x(pos,2) - 1;

plotClassData3SS(x,y,g)




g(x(:,2)>1) = 1;
g(x(:,2)<-1) = 1;



xl = xlim;
yl = ylim;

[X,Y] = meshgrid(xl(1):0.1:xl(2),yl(1):0.1:yl(2));
testx = [reshape(X,prod(size(X)),1) reshape(Y,prod(size(Y)),1)];

gam = 1;
testC = kernel(testx,x,'gauss',gam);

% Do some sampling
N = size(x,1);
C = kernel(x,x,'gauss',gam) + 1e-6*eye(N);
Ci = inv(C);

postC = inv(Ci + eye(N));


for a = [0 0.5]

    close all
    
    testprobs = zeros(size(testx,1),3);
    testvar = zeros(size(testx,1),1);
    Cprod = diag(testC*(C\testC'));
    testvar = repmat(1,size(testx,1),1) - Cprod;


    f = zeros(N,1);
    f = gausssamp(repmat(0,N,1),C,1)';
    z = zeros(N,1);
    NITS = 200;
    BURN = 100;
    tempy = y;
    for it = 1:NITS
        fprintf('Iteration: %g\n',it);
        for n = 1:N
            if g(n) == 1
                probm = normcdf(-a-f(n));
                probm = probm / ...
                    (probm + normcdf(f(n)-a));
                if rand<probm
                    % Sample from the negative truncation
                    tempy(n) = -1;
                else
                    tempy(n) = 1;
                end
            end
            z(n) = f(n) + randn;
            while z(n)*tempy(n)<a
                z(n) = f(n) + randn;
            end
        end
        f = gausssamp(postC*z,postC,1)';

        % Do the regression for predictions
        testmu = (testC/C)*f;

        testsample = randn(size(testx,1),1).*sqrt(testvar) + testmu;
        testz = testsample + randn(size(testx,1),1);
        if it>BURN
            pos = find(testz < -a);
            testprobs(pos,1) = testprobs(pos,1) + 1;
            pos = find(testz > a);
            testprobs(pos,3) = testprobs(pos,3) + 1;        
            pos = find(testz > -a & testz < a);
            testprobs(pos,2) = testprobs(pos,2) + 1;        

        end



    end

    testprobs = testprobs./repmat(sum(testprobs,2),1,3);
    close all
    surf(X,Y,reshape(testprobs(:,1),size(X)));
    view([0 0 1]);
    hold on
    plotClassData3SS(x,y,g);
    axis tight
    setupPlot;
    fname = sprintf('ncnm_a%g_g%g.eps',10*a,gam);
    makePDFopengl(fname);
end