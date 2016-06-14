function [ y_dzdx ] = vl_nnlogsigm( layer,resi,dzdy )
%VL_NNLOGSIGM Summary of this function goes here
%   Detailed explanation goes here
if nargin <3
    y_dzdx = 1./ (1 + exp(-sign(resi.x).*log(abs(resi.x)+1)));
else
    y_dzdx = (sign(resi.x)./((abs(resi.x)+2).^2)).*dzdy;
end

end

