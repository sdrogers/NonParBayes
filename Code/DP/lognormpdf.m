function l = lognormpdf(x,m,p)

l = -0.5*sqrt(2*pi) + 0.5*log(p) - 0.5*p*(x-m).^2;