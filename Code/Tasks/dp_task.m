% dp_task.m
clear all;
close all;
% Sample some data from a DP prior
N = 100; % number of data points to generate
base_mean = 0; % prior mean
base_prec = 1e-2; % prior preciision
obs_prec = 1; % observation noise precision
counts = [];
components = [];
K = 0;
alp = 1; % bigger alp = more clusters
for n = 1:N
    Prior = [counts alp];
    Prior = Prior./sum(Prior);
    thisComponent = find(rand<=cumsum(Prior),1);
    if thisComponent > K
        % Make a new component
        K = K + 1;
        components(1,K) = randn./sqrt(base_prec) + base_mean;
        counts(1,K) = 1;
        Z(n,1) = K;
        x(n,1) = randn./sqrt(obs_prec) + components(K);
    else
        Z(n,1) = thisComponent;
        x(n,1) = randn./sqrt(obs_prec) + components(thisComponent);
        counts(thisComponent) = counts(thisComponent) + 1;
    end
end

close all;
figure
hold all
for k = 1:K
    pos = find(Z==k);
    plot(x(pos),repmat(0,size(pos)),'o','markersize',10);
end

%% Posterior sampling
close all
x = sort(x);
N = length(x);
Z = repmat(1,N,1); % start everything in the same cluster
clustN = sum(N); % maintain the necessary values for the clusters
clustS = sum(x);
K = 1;

NITS = 1000;

alp = 1; % try varying this

simMatrix = zeros(N); % posterior similarity matrix
nClusters = []; % keep track of the number of clusters
samplePars = 0; % if 0, explicitly sample the parameters

if samplePars
    theta = [];
    [m,p] = getPost(clustN,clustS,base_mean,base_prec,obs_prec);
    theta = randn(1,K)./sqrt(p) + m;
end

for it = 1:NITS
    for n = 1:N % loop over objects
        current = Z(n);
        Z(n) = -1;
        % remove from its cluster
        clustN(current) = clustN(current) - 1;
        clustS(current) = clustS(current) - x(n);
        if clustN(current) == 0 % if cluster is now empty, delete
            pos = find(Z>current);
            Z(pos) = Z(pos) - 1;
            clustN(current) = [];
            clustS(current) = [];
            K = K - 1;
            if samplePars
                theta(current) = [];
            end
        end
        % Compute likelihood wrt clusters
        Like = zeros(1,K+1);
        for k = 1:K
            if samplePars % using parameter values
                Like(k) = lognormpdf(x(n),theta(k),obs_prec);
            else % marginalising cluster parameters
                Like(k) = clusterlike(x(n),clustN(k),clustS(k),base_mean,base_prec,obs_prec);
            end
        end
        
        % Add the likelihood of a new one (has to be marginalised)
        Like(end) = clusterlike(x(n),0,0,base_mean,base_prec,obs_prec);
        
        % Combine prior and likelihood
        Prior = [clustN alp]./(alp + N -1);
        Post = Like + log(Prior);
        Post = exp(Post - max(Post));
        Post = cumsum(Post./sum(Post));
        
        % Sample a new cluster index for this object
        new_cluster = find(rand<Post,1);
        if new_cluster > K % If we sampled a new index, make a new cluster
            K = K + 1;
            clustN(1,end+1) = 1;
            clustS(1,end+1) = x(n);
            Z(n) = max(Z) + 1;
            if samplePars
                [m,p] = getPost(1,x(n),base_mean,base_prec,obs_prec);
                theta(1,end+1) = randn/sqrt(p) + m;
            end
        else % add to the current cluster
            clustN(new_cluster) = clustN(new_cluster) + 1;
            clustS(new_cluster) = clustS(new_cluster) + x(n);
            Z(n) = new_cluster;
        end
    end
    if samplePars %if we are sampling the parameters, do it
        [m,p] = getPost(clustN,clustS,base_mean,base_prec,obs_prec);
        theta = randn(1,K)./sqrt(p) + m;
    end
    
    % Store the number of clusters
    nClusters(it) = K;

    
    for k = 1:K % update posterior similarity matrix
        pos = find(Z==k);
        simMatrix(pos,pos) = simMatrix(pos,pos) + 1;
    end
    if rem(it,10)==0 % do some visualisation every 10 samples
        figure(1);
        imagesc(simMatrix);drawnow
        figure(2);
        hold off
        for k = 1:K
            pos = find(Z==k);
            plot(x(pos),zeros(size(pos)),'o','markersize',20);
            hold all
        end
        figure(3)
        hist(nClusters,[1:10])
    end
    
end

