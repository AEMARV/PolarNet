function [ dzdw_masked,Masked ] = maskgrad( dzdw )
%MASKGRAD Summary of this function goes here
%   Detailed explanation goes here
   SUM = sum(abs(dzdw(:)));
   absDzdw  = abs(dzdw);
   absprobdzdw =absDzdw ./SUM;
   CUMSUM = cumsum(absprobdzdw(:));
   RANDOM = gpuArray.rand([1,2]);
   maskedmin = CUMSUM >= min(RANDOM(:));
   maskedmax  = CUMSUM < max(RANDOM(:));
   Last = find(maskedmax);
   maskedmax(Last +1) = 1;
   masked =  maskedmin .* maskedmax;
   Masked = gpuArray.zeros(size(dzdw));
   Masked(logical(masked)) = 1;
   %fprintf('%d random variables included \n',numel(find(Masked)));
   dzdw_masked= dzdw;
   dzdw_masked(~Masked) = 0;
end
function rel_selfInf = calc_ent_all(probs)
    selfInf =(-probs .* log(probs));
    rel_selfInf = selfInf./sum(selfInf(:));
end
