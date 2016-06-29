function [resip1,mask ] = vl_nnstochrelu(resi,resip1,dzdy )
%STOCHRELU Summary of this function goes here
%   Detailed explanation goes here
   if nargin == 1
     probs = vl_nnsigmoid(resi);  
     mask = (gpuArray.rand(size(probs))<= probs);
      resip1 = mask .* resi;
      return;
   end
     
     
     if nargin >2
         if isa(dzdy,'gpuArray')
         %resip1 = dzdy .* (resi.x>=0) ; %% thats resi conceptually
         [probs,MAX] = calculateprobs(resi.x);
         probsprime = probs;
         probsprime(probsprime == 1) = 0;
         dydx = resip1.aux + resi.x .* sampler(probsprime ./resi.x);
         resip1 = dzdy .* dydx;
         else
             if ~dzdy  %% it is now isStochastic
                 probs = vl_nnsigmoid( resi.x);
                 resip1.x = probs .* resi.x;
             end
         end
     else
         
      
         
        probs = calculateprobs(resi.x);
     mask = sampler(probs);
     resip1.aux = mask;
     resip1.x = bsxfun(@times,mask ,resi.x);
     end


% second imp
% if nargin == 1
%     probs = vl_nnsigmoid(layer);
%     y_dzdx = layer .* probs;
%     return;
% end
% probs = vl_nnsigmoid(resi.x);
% y_dzdx = resi.x .* probs;
% 
% if nargin>2
%     y_dzdx = (probs + (resi.x .* (probs.*(1-probs)))).*dzdy;
% end
end
function out = sampler(prob)

    out = prob>gpuArray.rand(size(prob));


end
function [prob,MAX] = calculateprobs(x)
type = 'max';
dims = [3];
MAX = x;
for i = 1:numel(dims)
    MAX = max(MAX,[],dims(i));
end
prob = bsxfun(@rdivide, x, MAX);
end
