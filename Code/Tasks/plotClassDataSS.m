function plotClassDataSS(x,t,g)
u = unique(t);
hold on
symbs = {'ko','ks','kv'};
cols = {[1 1 1],[0.6 0.6 0.6],[0.3 0.3 0.3]};
for i = 1:length(u)
    pos = find(t==u(i) & g==0);
    plot(x(pos,1),x(pos,2),symbs{i},'markerfacecolor',cols{i},'markersize',20);
end

pos = find(g==1)
plot(x(pos,1),x(pos,2),'ko','markersize',5,'markerfacecolor','k');