function plotClassData(x,t)
u = unique(t);
figure
hold on
cols = {[1 1 1],[0.6 0.6 0.6]};
for i = 1:length(u)
    pos = find(t==u(i));
    plot(x(pos,1),x(pos,2),'ko','markerfacecolor',cols{i},'markersize',20);
end