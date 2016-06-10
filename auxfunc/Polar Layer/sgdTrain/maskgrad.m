function [ dzdw_masked ] = maskgrad( dzdw )
%MASKGRAD Summary of this function goes here
%   Detailed explanation goes here
   SUM = sum(abs(dzdw(:)));
   absDzdw  = abs(dzdw);
   absprobdzdw =absDzdw ./SUM;
   RANDOM = rand(size(dzdw));
   mask = absprobdzdw<RANDOM;
   dzdw_masked= mask.*dzdw;
end

