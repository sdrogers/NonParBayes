function l = clusterlike(x,top_N,top_S,base_mean,base_prec,obs_prec)

postprec = base_prec + top_N*obs_prec;
postmean = (1/postprec)*(obs_prec*top_S + base_mean*base_prec);
predprec = 1/(1/postprec + 1/obs_prec);
l = lognormpdf(x,postmean,predprec);

