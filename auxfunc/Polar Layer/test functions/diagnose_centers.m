function [ dzdx0,dzdy0 ] = calcGradCenter( orIm, x0y0, dzdg)
% claculates the SGD on the center of attention.
% function [ dzdx0,dzdy0 ] = calcGradCenter( orIm, x0y0, dzdg)
%
%
x0  = x0y0(1,:);
y0 = x0y0(2,:);
[dfdy, dfdx] = gradient(orIm);
% g(r,t) = f(rcost + x0 ,rsint + y0)
% dg(r,t)dx0 = df/dx0 = 1: d(rcost + x0)/dx0 * df(rcost+x0,rsint+y0);
end

function dgdx0y0 = getslope(orIm,x0y0)