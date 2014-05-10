function C = kernel(X,Y,type,par)

switch type
    case 'linear'
        C = X*Y';
    case 'gauss'
        nx = size(X,1);
        ny = size(Y,1);
        distance=sum((X.^2),2)*ones(1,ny) +...
         ones(nx,1)*sum((Y.^2),2)' - 2*(X*Y');
        C = exp(-0.5*par(1)*distance);
        if length(par)==2
            C = C *par(2);
        end
    case 'cosine'
        C = X*Y';
        sX = sqrt(sum(X.^2,2));
        sY = sqrt(sum(Y.^2,2));
        C = C./(repmat(sX,1,size(Y,1)).*...
            repmat(sY',size(X,1),1));
    case 'diag'
        C = eye(size(X,1)); % This one can't do any testing
    case 'poly'
        sX = sqrt(sum(X.^2,2));
        sY = sqrt(sum(Y.^2,2));
        X = X./repmat(sX,1,size(X,2));
        Y = Y./repmat(sY,1,size(Y,2));
        C = (1+X*Y').^par;
    case 'rq'
        nx = size(X,1);
        ny = size(Y,1);
        distance=sum((X.^2),2)*ones(1,ny) +...
         ones(nx,1)*sum((Y.^2),2)' - 2*(X*Y');
        C = (1 + distance./(2*par(1)*par(2)^2));
    case 'nn'
        g = par(2)^(-2);
        Gii = g*(1+repmat(X,1,length(Y)).^2);
        Gjj = g*(1+repmat(Y',length(X),1).^2);
        Gij = g*(1+repmat(X,1,length(Y)).*repmat(Y',length(X),1));
        C = par(1)*asin(Gij./(((1+Gii).*(1+Gjj)).^(1/2)));
    case 'nnd'
    g = par(2)^(-2);
        Gii = g*(1+repmat(X,1,length(Y)).^2);
        Gjj = g*(1+repmat(Y',length(X),1).^2);
        Gij = g*(1+repmat(X,1,length(Y)).*repmat(Y',length(X),1));
        Xr = repmat(X,1,length(Y));
        Yr = repmat(Y',length(X),1);
    C = par(1)*((-Yr+Xr+(1/g)*Xr).*(1+(1/g)+Xr.^2))./...
        (((1+(1/g)+Yr.^2).*(1+(1/g)+Xr.^2)).*...
        (1/g^2 + (Xr-Yr).^2 + (1/g)*(2+Yr.^2+Xr.^2)).^(1/2));
%         C = g*par(1)*(((1+Gii).*(1+Gjj) - Gij.^2).^(-1/2)).*...
%             (Xr - Gij.*Yr.*(1./(1+Gjj)));
    case 'nndd' % Derivate of nn
        g = par(2)^(-2);
        Gii = g*(1+repmat(X,1,length(Y)).^2);
        Gjj = g*(1+repmat(Y',length(X),1).^2);
        Gij = g*(1+repmat(X,1,length(Y)).*repmat(Y',length(X),1));
        Xr = repmat(X,1,length(Y));
        Yr = repmat(Y',length(X),1);
        C = par(1)*(2+1/g)./...
            (g*(1/g^2 + (Yr-Xr).^2 + (1/g)*(2+Xr.^2 + Yr.^2)).^(3/2));
%         C = g*par(1)*((((1+Gii).*(1+Gjj)-Gij.^2).^(-1/2)).*...
%             (1-g*Xr.^2.*(1./(1+Gii)))-...
%             (0.5*(Yr - Gij.*Xr.*(1./(1+Gii))).*...
%             (((1+Gii).*(1+Gjj) - Gij.^2).^(-3/2)).*...
%             (2*g.*Yr.*(1+Gii) - 2*g*Gij.*Xr)));
        
end
