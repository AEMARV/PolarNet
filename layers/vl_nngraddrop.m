function [ y_dzdx ] = vl_nngraddrop(layer,resi,dzdy )
%VL_GRADDROP Summary of this function goes here
%   Detailed explanation goes here
    if nargin <3
        y_dzdx = resi.x;
    else
        y_dzdx = maskgrad(dzdy);
    end

end

