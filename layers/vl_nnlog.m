function y_dzdx = vl_nnlog(layer,resi,dzdy )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin <3
    y_dzdx = sign(resi.x).*log(abs(resi.x)+1);
else
    y_dzdx = (1/(1+abs(resi.x))) .*  dzdy;
end

end

