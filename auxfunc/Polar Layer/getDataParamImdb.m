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
    DataParam = DataParam(batch);
    Fnames = fieldnames(DataParam);
    cellParam = struct2cell(DataParam);
    cellParam_gpu = cellfun(@gpuArray,cellParam);
    if flip
        % change col0 = - col0
        % change theta0 = -theta0
        Ind = strcmp(Fnames,'col0') |strcmp(Fnames,'theta0') ;
        
        cellParam_gpu(Ind,:,:) = cellfun(@uminus,cellParam_gpu,'UniformOutput',false);
        
        
    end
DataParam = cell2struct(cellParam_gpu,Fnames,1);
end

