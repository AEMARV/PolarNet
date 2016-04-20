function normalized = sigmoid(preds,type)
    % calculates the softmax of preds meaning simoid(preds)/sum(sigmoid(preds))
% function normalized = sigmoid(preds)
    if strcmp(type,'sigmoid')
    MAX = max(preds,[],3);
    normalized = bsxfun(@minus,preds,MAX);
    normalized = exp(normalized);
    normalized = bsxfun(@rdivide,normalized, sum(normalized,3));
    else
        if strcmp(type,'linear')
            normalized = bsxfun(@minus,preds,min(preds));
        end
    end
    
end