function [ resip1 ] = vl_nnstochactive( resi,resip1,dzdy,isStochastic )
%VL_NNSTOCHACTIVE Summary of this function goes here
%   Detailed explanation goes here
if isempty(isStochastic)
isStochastic = true;    
end

if nargin > 2 && ~isempty(dzdy) ; doder = true;else doder = false;end
if ~doder
    % forward pass
if isStochastic
    
in = resi.x;
probs = calculateprobs(in);
mask = sampler(probs);
resip1.x =  bsxfun(@times,in,mask);
resip1.aux = mask;

else
    
end
else
    [probs,MAX] = calculateprobs(resi.x);
         probsprime = probs;
         probsprime(probsprime == 1) = 0;
         dydx = sampler(probs) + resi.x .* sampler(probsprime ./resi.x);
         resip1 = dzdy .* dydx;
end

end

function out = sampler(prob)

    out = prob>gpuArray.rand(size(prob));


end
function [prob,MAX] = calculateprobs(x)
type = 'max';
dims = 3;
MAX = x;
for i = 1:numel(dims)
    MAX = max(MAX,[],dims(i));
end
prob = bsxfun(@rdivide, x, MAX);
end
