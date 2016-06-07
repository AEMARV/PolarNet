function y_dzdx = vl_nnlog(layer,resi,dzdy )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin <3
    y_dzdx = log(resi.x);
else
    y_dzdx = (1 ./resi.x) .*  dzdy;
end

end

