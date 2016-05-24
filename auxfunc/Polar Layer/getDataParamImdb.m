function [DataParam, imdb ] = getDataParamImdb( imdb,batch,flip )
%% function [DataParam, imdb ] = getDataParamImdb( imdb,batch,flip )
% gets the centers from the imdb according to the batch array and returns
% the gpuArray centers. !!!! centers are gpuArray !!!
%
% INPUT :
% the only notable input is flip which tells the function to obtain the
% lr_fliped version of centers
% you have to keep this flag when you want to update the centers with
% whatever method you are using. since the images in the imdb does not
% change the centers also should return to the original orientation before
% saving them.
%

DataParam = imdb.DataParam;
DataParam = getParamBatch(DataParam,batch);

if flip
    % change col0 = - col0
    % change theta0 = -theta0
    DataParam.col0 = -DataParam.col0;
    DataParam.theta0 = - DataParam.theta0;
end
end
function DataParam = getParamBatch(DataParam,batch)
Fnames = fieldnames(DataParam);
for i = 1:numel(Fnames)
    DataParam.(Fnames{i}) = gpuArray(DataParam.(Fnames{i})(batch,:));
end

end

