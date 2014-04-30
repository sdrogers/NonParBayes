%% DP Script
path(path,'../');
clear all;
close all;

% Sample from the fixed model
N = 100;
K = 20;

alpvals = [0.1 1 10 100];

NITS = 1000;
for a = 1:length(alpvals)
    Z = zeros(N,K);
    Z(:,1) = 1;
    alp = alpvals(a);
    for it = 1:NITS
        for n = 1:N
            Z(n,:) = 0;
            c = sum(Z);
            probs = (c + alp/K)./(alp + sum(c));
            pos = find(rand<=cumsum(probs));
            Z(n,pos) = 1;
        end
        c = sum(Z);
        nz(it,a) = sum(c>0);
    end
end
%%
b = boxplot(nz)
for i = 1:length(b(:))
    set(b(i),'linewidth',2);
end
set(gca,'xtick',[1:4],'xticklabel',alpvals)

setupPlot;

xlabel('$\alpha$','interpreter','latex','fontsize',24)
ylabel('Number of non-empty components');
makePDF('DPfixed.eps');