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

%% Sample from DP prior
close all;
N = 100;
alpvals = [0.1 1 10 100];
nClusters = [];
for a = 1:length(alpvals)
    Z = repmat(1,N,1);
    cluster_counts = N;
    alp = alpvals(a);
    for it = 1:NITS
        for n = 1:N
            current = Z(n);
            Z(n) = -1;
            cluster_counts(current) = cluster_counts(current) - 1;
            if cluster_counts(current) == 0
                % Delete
                cluster_counts(current) = [];
                pos = find(Z>current);
                Z(pos) = Z(pos) - 1;
            end

            Post = [cluster_counts alp];
            Post = Post./sum(Post);
            new_cluster = find(rand<=cumsum(Post),1);

            if new_cluster > max(Z)
                cluster_counts(1,end+1) = 1;
                Z(n) = max(Z) + 1;
            else
                cluster_counts(new_cluster) = cluster_counts(new_cluster) + 1;
                Z(n) = new_cluster;
            end

        end
        nClusters(it,a) = length(cluster_counts);
    end
end

%%
close all
hist(nClusters,1:2:max(nClusters(:)))
setupPlot
xlabel('Number of non-empty clusters');
ylabel('Frequency');
legend('\alpha = 0.1','\alpha = 1','\alpha = 10','\alpha = 100')
axis tight
makePDF('DPbars.eps');

%% DP Mixture example
% Sample some nice data
x = [randn(50,1)-5;randn(50,1);randn(50,1)+5];
x = sort(x);
base_mean = 0;
base_prec = 0.01;
obs_prec = 1;

N = length(x);
Z = repmat(1,N,1);
clustN = sum(N);
clustS = sum(x);
K = 1;
alp = 1;
simMatrix = zeros(N);
nClusters = [];
samplePars = 0;

if samplePars
    theta = [];
    [m,p] = getPost(clustN,clustS,base_mean,base_prec,obs_prec);
    theta = randn(1,K)./sqrt(p) + m;
end
for it = 1:NITS
    for n = 1:N
        current = Z(n);
        Z(n) = -1;
        clustN(current) = clustN(current) - 1;
        clustS(current) = clustS(current) - x(n);
        if clustN(current) == 0
            % Delete
            pos = find(Z>current);
            Z(pos) = Z(pos) - 1;
            clustN(current) = [];
            clustS(current) = [];
            K = K - 1;
            if samplePars
                theta(current) = [];
            end
        end
        
        Like = zeros(1,K+1);
        for k = 1:K
            if samplePars
                Like(k) = lognormpdf(x(n),theta(k),obs_prec);
            else
                Like(k) = clusterlike(x(n),clustN(k),clustS(k),base_mean,base_prec,obs_prec);
            end
        end
        Like(end) = clusterlike(x(n),0,0,base_mean,base_prec,obs_prec);
        Prior = [clustN alp]./(alp + N -1);
        Post = Like + log(Prior);
        Post = exp(Post - max(Post));
        Post = cumsum(Post./sum(Post));
        new_cluster = find(rand<Post,1);
        if new_cluster > K
            % create a new one
            K = K + 1;
            clustN(1,end+1) = 1;
            clustS(1,end+1) = x(n);
            Z(n) = max(Z) + 1;
            if samplePars
                [m,p] = getPost(1,x(n),base_mean,base_prec,obs_prec);
                theta(1,end+1) = randn/sqrt(p) + m;
            end
        else
            clustN(new_cluster) = clustN(new_cluster) + 1;
            clustS(new_cluster) = clustS(new_cluster) + x(n);
            Z(n) = new_cluster;
        end
    end
    if samplePars
        [m,p] = getPost(clustN,clustS,base_mean,base_prec,obs_prec);
        theta = randn(1,K)./sqrt(p) + m;
    end
    for k = 1:K
        pos = find(Z==k);
        simMatrix(pos,pos) = simMatrix(pos,pos) + 1;
    end
    if rem(it,10)==0
        figure(1);
        imagesc(simMatrix);drawnow
        figure(2);
        hold off
        for k = 1:K
            pos = find(Z==k);
            plot(x(pos),zeros(size(pos)),'o','markersize',20);
            hold all
        end
    end
    
    nClusters(it) = K;
end
%%
close all
figure(1);
hold off
for n = 1:N
    plot([x(n) x(n)],[0 1],'k');
    hold on
    plot(x(n),1,'ro')
end
ylim([0 2])
setupPlot
xlabel('x');
makePDF('CRP_data.eps');

figure(2)
imagesc(simMatrix);
setupPlot;
xlabel('n');
ylabel('n');
makePDFopengl('CRP_sim.eps');

figure(3);
hist(nClusters,unique(nClusters));
setupPlot;
xlabel('K');
ylabel('Frequency');
makePDF('CRP_K.eps');


%% Sample from a HDP


