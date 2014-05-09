% Gibbs sampling for a hierarchical DP
function hdp_sample(x)
% Data must be stored in x{}

NITS = 1000;

base_mean = 0;
base_prec = 1e-6;

obs_prec = 1;


alpha = 1;
gamma = 1;

J = length(x);



% initialise everything in one cluster
top_N = 0;
top_S = 0;
N_top = 1;
N_tables_per_top = J;
for j = 1:J
    N(j) = length(x{j});
    tables_to_top{j} = [1];
    Z{j} = repmat(1,N(j),1);
    top_N = top_N + N(j);
    top_S = top_S + sum(x{j});
    N_tables(j) = 1;
    table_sums{j} = N(j);
end

allN_top = [1];


for it = 1:NITS
    jorder = randperm(J);
    for j = 1:J
        thisj = jorder(j);
        norder = randperm(N(thisj));
        for n = 1:N(thisj)
            thisn = norder(n);
            
            % Remove thisn from the model
            thistable = Z{thisj}(thisn);
            Z{thisj}(thisn) = -1;
            thistop = tables_to_top{thisj}(thistable);
            table_sums{thisj}(thistable) = table_sums{thisj}(thistable) - 1;
            top_N(thistop) = top_N(thistop) - 1;
            top_S(thistop) = top_S(thistop) - x{thisj}(thisn);
            
            if table_sums{thisj}(thistable) == 0
                % Delete the table
                table_sums{thisj}(thistable) = [];
                tables_to_top{thisj}(thistable) = [];
                N_tables(thisj) = N_tables(thisj) - 1;
                N_tables_per_top(thistop) = N_tables_per_top(thistop) - 1;
                pos = find(Z{thisj}>thistable);
                Z{thisj}(pos) = Z{thisj}(pos) - 1;
            end
            
            if N_tables_per_top(thistop) == 0
                % Delete this top level component
                N_tables_per_top(thistop) = [];
                if top_N(thistop) ~= 0 
                    keyboard
                end
                top_N(thistop) = [];
                top_S(thistop) = [];
                N_top = N_top - 1;
                for jp = 1:J
                    pos = find(tables_to_top{jp}>thistop);
                    tables_to_top{jp}(pos) = tables_to_top{jp}(pos) - 1;
                end
            end
            
            
            TableLike = zeros(1,N_tables(thisj)+1);
            for table = 1:N_tables(thisj)
                TableLike(table) = clusterlike(x{thisj}(thisn),top_N(tables_to_top{thisj}(table)),...
                    top_S(tables_to_top{thisj}(table)),base_mean,base_prec,obs_prec);
            end
            TablePrior = [table_sums{thisj} alpha]./(alpha + sum(table_sums{thisj}));
            
            
            newTableLike = zeros(1,N_top+1);
            for i = 1:N_top
                newTableLike(i) = clusterlike(x{thisj}(thisn),top_N(i),top_S(i),base_mean,base_prec,obs_prec);
            end
            newTableLike(end) = clusterlike(x{thisj}(thisn),0,0,base_mean,base_prec,obs_prec);
            newTablePrior = [N_tables_per_top gamma]./(gamma + sum(N_tables_per_top));
            
            TableLike(end) = log(sum(exp(newTableLike).*newTablePrior)); % The margilised bit
            
            TablePost = TableLike + log(TablePrior);
            TablePost = exp(TablePost - max(TablePost));
            TablePost = cumsum(TablePost./sum(TablePost));
            
            newTable = find(rand<=TablePost,1);
            
            if newTable < N_tables(thisj)
                % This is a current table
                Z{thisj}(thisn) = newTable;
                table_sums{thisj}(newTable) = table_sums{thisj}(newTable) + 1;
                newTop = tables_to_top{thisj}(newTable);
                top_N(newTop) = top_N(newTop) + 1;
                top_S(newTop) = top_S(newTop) + x{thisj}(thisn);
            else
                % This is a new table
                % Choose the component
                topPost = newTableLike + log(newTablePrior);
                topPost = exp(topPost - max(topPost));
                topPost = cumsum(topPost./sum(topPost));
                newTop = find(rand<=topPost,1);
                N_tables(thisj) = N_tables(thisj) + 1;
                if newTop < N_top
                    % New table with current component
                    tables_to_top{thisj}(1,end+1) = newTop;
                    Z{thisj}(thisn) = max(Z{thisj}) + 1;
                    table_sums{thisj}(1,end+1) = 1;
                    top_N(newTop) = top_N(newTop) + 1;
                    top_S(newTop) = top_S(newTop) + x{thisj}(thisn);
                    N_tables_per_top(newTop) = N_tables_per_top(newTop) + 1;
                else
                    % New table with new component
                    Z{thisj}(thisn) = max(Z{thisj}) + 1;
                    table_sums{thisj}(1,end+1) = 1;
                    N_top = N_top + 1;
                    top_N(1,N_top) = 1;
                    top_S(1,N_top) = x{thisj}(thisn);
                    tables_to_top{thisj}(1,end+1) = N_top;
                    N_tables_per_top(N_top) = 1;
                end
                
            end
            
        end
        figure(1);
        subplot(1,J,thisj),plot(Z{thisj});drawnow;
        checkModel(x,top_N,top_S,Z,tables_to_top,N_tables_per_top,N_top,N_tables);
    end
    allN_top = [allN_top N_top];
    figure(2);
    plot(allN_top);drawnow;
    
    % Get the top component means
    postprec = base_prec + top_N*obs_prec;
    postmean = (1./postprec).*(obs_prec.*top_S + base_mean*base_prec);
    postmean
    
end

function l = clusterlike(x,top_N,top_S,base_mean,base_prec,obs_prec)

postprec = base_prec + top_N*obs_prec;
postmean = (1/postprec)*(obs_prec*top_S + base_mean*base_prec);
predprec = 1/(1/postprec + 1/obs_prec);
l = lognormpdf(x,postmean,predprec);

function l = lognormpdf(x,m,p)

l = -0.5*sqrt(2*pi) + 0.5*log(p) - 0.5*p*(x-m).^2;

function o = checkModel(x,top_N,top_S,Z,tables_to_top,N_tables_per_top,N_top,N_tables)

J = length(x);

if N_top ~= length(top_N)
    keyboard
end

if N_top ~= length(top_S)
    keyboard
end

new_t_t_t = zeros(1,N_top);
for i = 1:N_top
    for j = 1:J
        new_t_t_t(i) = new_t_t_t(i) + sum(tables_to_top{j}==i);
    end
end

if sum(new_t_t_t ~= N_tables_per_top) > 0
    keyboard
end

new_top_N = zeros(1,N_top);
new_top_S = zeros(1,N_top);
for j = 1:J
    for n = 1:length(x{j})
        thistable = Z{j}(n);
        thistop = tables_to_top{j}(thistable);
        new_top_N(thistop) = new_top_N(thistop) + 1;
        new_top_S(thistop) = new_top_S(thistop) + x{j}(n);
    end
end

if sum(new_top_N ~= top_N) > 0
    keyboard
end

if sum(abs(new_top_S-top_S)>1e-9) > 0
    keyboard
end

o =1;
