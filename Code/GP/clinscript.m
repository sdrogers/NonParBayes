%% Clinical script
clear all;
%% Generate some data
close all;
x = [0:0.1:10]';
N = length(x);
C = 2*kernel(x,x,'gauss',1) + 1e-6*eye(N);;
b = [-inf -3 -1 1 3 inf];
f = gausssamp(repmat(0,N,1),C,1)';
plot(x,f,'k','linewidth',2);
setupPlot;
xlabel('t');
ylabel('f');
ylim([-4.5 4.5])

makePDF('health.eps');

C = 3;
delta = [0 -1 2];
prec = [0.5 5 10];
q = randn(N,C).*repmat(sqrt(1./prec),N,1) + repmat(f,1,C) + repmat(delta,N,1);
hold all
cols = {'r','b','g'};
for c = 1:C
    plot(x,q(:,c),[cols{c} 'o']);
end
ylim([-4.5 4.5])

makePDF('health_corrupted.eps');

for i = 2:length(b)-1
    plot(xlim,[b(i) b(i)],'k--');
end

y = zeros(size(q));
pv = [-4 -2 0 2 4];
for c = 1:C
    for i = 1:length(b)-1
        y(q>b(i) & q<b(i+1)) = pv(i);
    end
end
for c = 1:C
    plot(x,y(:,c),cols{c},'linewidth',2)
end
set(gca,'ytick',pv,'yticklabel',{'A','B','C','D','E'});
ylim([-4.5 4.5])
makePDF('health_corrupted_ratings.eps');