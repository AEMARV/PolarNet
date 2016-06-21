function [resip1 ] = vl_nnstochrelu(resi,resip1,dzdy )
%STOCHRELU Summary of this function goes here
%   Detailed explanation goes here
   if nargin == 1
     probs = vl_nnsigmoid(resi);  
     mask = (gpuArray.rand(size(probs))<= probs);
      resip1 = mask .* resi;
      return;
   end
     
     
     if nargin >2
         resip1 = dzdy .* resip1.aux; %% thats resi conceptually
     else
         
         probs = vl_nnsigmoid( resi.x);
     mask = (gpuArray.rand(size(probs))<= probs);
     resip1.aux = mask;
     resip1.x = mask .* resi.x;
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

