function plotClassData3(x,t)
u = unique(t);
hold on
symbs = {'ko','ks','kv'};
cols = {[1 1 1],[0.6 0.6 0.6],[0.3 0.3 0.3]};
for i = 1:length(u)
    pos = find(t==u(i));
    plot3(x(pos,1),x(pos,2),repmat(1,length(pos),1),symbs{i},'markerfacecolor',cols{i},'markersize',20);
end