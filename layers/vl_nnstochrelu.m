function [y_dzdx] = vl_nnstochrelu(x,dzdy )
prob = vl_nnsigmoid(x);
if nargin < 2
    prob = sampler(prob);
    y_dzdx = prob.*x;
else
    y_dzdx = dzdy.*(sampler(prob) + x.*(sampler(prob)- sampler(prob).*(sampler(prob))));
end
end
