function plotClassData3(x,t)
u = unique(t);
hold on
cols = {[1 1 1],[0.6 0.6 0.6]};
for i = 1:length(u)
    pos = find(t==u(i));
    plot3(x(pos,1),x(pos,2),repmat(1,length(pos),1),'ko','markerfacecolor',cols{i},'markersize',20);
end