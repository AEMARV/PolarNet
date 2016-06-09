function y_dzdx = vl_nnexp(layer,resi,dzdy)
if nargin < 3
    y_dzdx = sign(resi.x).*(exp(abs(resi.x))-1);
    
else
    y_dzdx = exp(abs(resi.x)).*dzdy;
end
