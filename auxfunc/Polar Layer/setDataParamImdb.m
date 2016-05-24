function imdb = setDataParamImdb( imdb,DataParam,Batch,isFlip )
%% function imdb = setDataParamImdb( imdb,DataParam,Batch,isFlip )
% transfers DataParam to cpu ram and then sets the data in imdb to
% DataParam according to Batch indices. 
% if isFlip is true it will negate theta0 and col0 in DataParam struct
    
    if isFlip
        DataParam.col0 = - DataParam.col0;
        DataParam.theta0 = - DataParam.theta0;
    end
    imdb.DataParam = setParamBatch(imdb.DataParam,DataParam,Batch);

end

function DataParamWhole = setParamBatch(DataParamWhole,DataParam,batch)
Fnames = fieldnames(DataParam);
for i = 1:numel(fieldnames(DataParam))
    DataParamWhole.(Fnames{i})(batch,:) = gather(DataParam.(Fnames{i}));
end

end

