function Entropy = AM_entropy( class_pred,isNormalized )
%SOFTMAX Summary of this function goes here
%   Detailed explanation goes here
class_pred = bsxfun(@rdivide, class_pred,sum(class_pred,3));
Entropy = class_pred .* log2(class_pred);
Entropy(isnan(Entropy)) = 0;
Entropy = -sum(Entropy,3);
if isNormalized
Entropy = Entropy/log2(size(class_pred,3));
end
end

