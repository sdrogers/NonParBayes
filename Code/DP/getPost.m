function [m,p] = getPost(N,S,mu0,gamma0,gamma)

p = gamma0 + N.*gamma;
m = (1./p).*(mu0*gamma0 + gamma.*S);