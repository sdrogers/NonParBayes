function makePDFopengl(epsname)
print('-depsc','-opengl',epsname);
eps2pdf(epsname,'/usr/local/bin/gs');