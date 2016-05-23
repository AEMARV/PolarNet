function imdb = setDataParamImdb( imdb,DataParam,Batch,isFlip )
%% function imdb = setDataParamImdb( imdb,DataParam,Batch,isFlip )
% transfers DataParam to cpu ram and then sets the data in imdb to
% DataParam according to Batch indices. 
% if isFlip is true it will negate theta0 and col0 in DataParam struct
    Fnames = fieldnames(DataParam);
    DataParamCell_gpu = struct2cell(DataParam);
    if isFlip
        fieldInd = strcmp(Fnames,'col0') | strcmp(Fnames,'theta0');
        DataParamCell_gpu(fieldInd,:,:) = cellfun(@uminus,DataParamCell_gpu(fieldInd,:,:),'UniformOutput',false);
    end
    DataParamCell_cpu = cellfun(@gather,DataParamCell_gpu,'UniformOutput',false);
    DataParam_cpu = cell2struct(DataParamCell_cpu,Fnames,1);
    imdb.DataParam(Batch) = DataParam_cpu;

end

