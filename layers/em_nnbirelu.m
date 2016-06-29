function [ y_dzdx ] = em_nnbirelu(x,dzdy )
%EM_NNBIRELU Summary of this function goes here
%   Detailed explanation goes here
isStoch = false;
chSize = size(x,3);
xcat = cat(3,x,-x);
if nargin<2
    if isStoch
    y_dzdx = vl_nnstochrelu(xcat);
    else
        y_dzdx = vl_nnrelu(xcat);
    end
else
    if isStoch
    y_dzdx = vl_nnstochrelu(xcat,dzdy);
    else
    y_dzdx = vl_nnrelu(xcat,dzdy);
    end
    y_dzdx = y_dzdx(:,:,1:end/2,:) - y_dzdx(:,:,(end/2)+1:end,:);
end

end

