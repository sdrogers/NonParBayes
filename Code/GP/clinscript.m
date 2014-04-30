%% Clinical script
clear all;
%% Generate some data
close all;
x = [0:0.01:10]';
C = kernel(x,x,'gauss',1);
b = [-inf -3 -1 1 3 inf];
N = length(x);
f = gausssamp(repmat(0,N,1),C,1)';