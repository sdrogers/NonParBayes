% gp_ncnm_task.m
data = twospirals(400,100,5,100.0);
y = data(:,3);x = data(:,1:2);
y(y==0) = -1;
N = size(y,1);

a = 0.5;

% 'hide' some data
g = zeros(N,1);
g(x(:,2)>1) = 1;
g(x(:,2)<-1) = 1;

if a==0
    pos = find(g==1);
    x(pos,:) = [];
    y(pos) = [];
    g(pos) = [];
end

% Make a grid for the test points
xl = [-1 1];
yl = [-2 2];
[X,Y] = meshgrid(xl(1):0.1:xl(2),yl(1):0.1:yl(2));
testx = [reshape(X,prod(size(X)),1) reshape(Y,prod(size(Y)),1)];

gam = 1;
testC = kernel(testx,x,'gauss',gam);

% Do some sampling
N = size(x,1);
C = kernel(x,x,'gauss',gam) + 1e-6*eye(N);
Ci = inv(C);

postC = inv(Ci + eye(N));






testprobs = zeros(size(testx,1),3);
testvar = zeros(size(testx,1),1);
Cprod = diag(testC*(C\testC'));
testvar = repmat(1,size(testx,1),1) - Cprod;


f = zeros(N,1);
f = gausssamp(repmat(0,N,1),C,1)';
z = zeros(N,1);
NITS = 300;
BURN = 100;
tempy = y;

for it = 1:NITS
    fprintf('Iteration: %g\n',it);
    for n = 1:N
        if g(n) == 1
            % If this is a 'semi' point, sample from the mixture
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
plotClassDataSS(x,y,g);
hold on
[c,h] = contour(X,Y,reshape(testprobs(:,1),size(X)));
clabel(c,h);
axis tight

