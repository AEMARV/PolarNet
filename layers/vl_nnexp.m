function y_dzdx = vl_nnexp(layer,resi,dzdy)
if nargin < 3
    y_dzdx = exp(resi.x);
else
    y_dzdx = exp(resi.x) .* dzdy;
end
