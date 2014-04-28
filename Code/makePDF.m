function makePDF(epsname)
print('-depsc',epsname);
eps2pdf(epsname,'/usr/local/bin/gs');