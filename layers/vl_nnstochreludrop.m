function [resip1,mask ] = vl_nnstochreludrop(resi,resip1,dzdy )
%STOCHRELU Summary of this function goes here
%   Detailed explanation goes here
LR = 0.01;
gateStart = (size(resi.x,3)/2)+1 ; 
gateInds = gateStart :size(resi.x,3);
XInds = 1 : gateStart -1;
if isempty(dzdy)
    doder = false;
else
    doder = true;
end
if ~doder
    in = resi.x;
    gate = in(:,:,gateInds,:);
    X = (in(:,:,XInds,:));
    probs = vl_nnsigmoid(gate);
    Activate = sampler(probs);
    resip1.x = bsxfun(@times,Activate,vl_nnrelu(X));
    resip1.aux = Activate;
else
    X = resi.x(:,:,XInds,:);
    gate = resi.x(:,:,gateInds,:);
    resip1 = resi.x;
    probs = vl_nnsigmoid(gate);
    dzdprob = dzdy .* X + calcregder(probs)  ;
    dzdgate =  bsxfun(@times,dzdprob , (sampler(probs) - sampler(probs).*sampler(probs)));
    dzdreluedX = bsxfun(@times,dzdy , sampler(probs)) ;
    dzdX = vl_nnrelu(X,dzdreluedX);
    if numel(gateInds) == 1
      resip1(:,:,gateInds,:) = sum(dzdgate,3)*LR;  
    else
      resip1(:,:,gateInds,:) = dzdgate*LR;
    end
    resip1(:,:,XInds,:) = dzdX;
end
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
% end\
function dregdprob = calcregder(probs)
    S1 = size(probs,1);
    S2 = size(probs,2);
    S3 = size(probs,3);
    SUM_HW = sum(sum(probs,1),2);
    Area = S1 * S2;
    dregareadprob = sign((SUM_HW/Area)  - 0.5)/Area;
    SUM_Channel = sum(probs,3);
    dregChannel = sign((SUM_Channel/S3) - 0.5)/S3;
    dregdprob = bsxfun(@plus,dregareadprob , dregChannel);
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
