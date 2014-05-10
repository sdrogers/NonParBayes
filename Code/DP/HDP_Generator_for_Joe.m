%% Generate some data from a HDP
base_mean = 0;
base_prec = 1e-2;

obs_prec = 1;

J = 5; % Number of files
N = 50; % 100 observations in all files
top_components = [];
top_counts = [];
nTop = 0;

alpha = 1; %Table DP
gamma = 1; % Component DP

for j = 1:J
    Z{j} = [];
    x{j} = [];
    cluster_counts{j} = [];
    tables_to_top{j} = [];
    nClusters{j} = 0;
    for n = 1:N
        % Choose a table
        probs = [cluster_counts{j} alpha];
        probs = cumsum(probs./sum(probs));
        table_choice = find(rand<=probs,1);
        if table_choice > nClusters{j}
            % Have to make a new table
            nClusters{j} = nClusters{j} + 1;
            Z{j}(n,1) = nClusters{j};
            cluster_counts{j}(nClusters{j}) = 1;
            
            % Choose a top level component for this new table
            probs = [top_counts gamma];
            probs = cumsum(probs./sum(probs));
            top_choice = find(rand<=probs,1);
            
            if top_choice > nTop
                % Create a new top level
                nTop = nTop + 1;
                top_counts(1,nTop) = 1;
                top_components(1,nTop) = randn./sqrt(base_prec) + base_mean;
                tables_to_top{j}(nClusters{j}) = nTop;
                x{j}(n,1) = randn./sqrt(obs_prec) + top_components(tables_to_top{j}(nClusters{j}));
            else
                % Use the current one
               tables_to_top{j}(nClusters{j}) = top_choice;
               top_counts(top_choice) = top_counts(top_choice) + 1;
               x{j}(n,1) = randn./sqrt(obs_prec) + top_components(tables_to_top{j}(nClusters{j}));
            end
           
        else
            % Add to the current table
            Z{j}(n,1) = table_choice;
            cluster_counts{j}(table_choice) = cluster_counts{j}(table_choice) + 1;
            x{j}(n,1) = randn./sqrt(obs_prec) + top_components(tables_to_top{j}(table_choice));
        end
    end
            
end


% Plot the data
close all
for j = 1:J
    [f,xi] = ksdensity(x{j});
    plot(xi,f);
    hold all
end


for j = 1:J
    [b,I] = sort(Z{j});
    x{j} = x{j}(I);
end

% Plot HDP data
close all
symbs = {'ro','k^','bs','g*','mv'};
for j = 1:J
    plot(x{j},j,symbs{j},'markersize',10);
    hold on
end

yl = ylim;
for t = 1:length(top_components)
    plot([top_components(t) top_components(t)],[yl],'k--','linewidth',2);
end

set(gca,'ytick',1:J)
setupPlot
xlabel('x');
ylabel('Dataset');
makePDF('hdp_data.eps');

%%
out = hdp_sample(x)
%%
close all
hist(out,[1:10]);
setupPlot
xlabel('Number of top-level components');
ylabel('Frequency')
makePDF('hdp_inference.eps');
