function [imdbRotated] = ImdbRotation(imdb,valsetNumber,batchsize)
%% 
% valsetNumber is the set index for validation set
% batchsize is the batch size in order to process rotation with gpu
    valIndex = find(imdb.images.set==valsetNumber);
    images = imdb.images.data;
    imdbRotated = imdb;
    for batchNumber = 1:size(valIndex,2)/batchsize
      batchSIndex = 1+(batchNumber-1)*batchsize;batchEIndex = batchNumber*batchsize;
      valBatchIndex = valIndex(batchSIndex:batchEIndex);
      valImages = gpuArray(images(:,:,:,valBatchIndex));
      theta = randi(359,1,batchsize);
      for index = 1:batchsize
          valImages(:,:,:,index)= imrotate(valImages(:,:,:,index),theta(index),'crop');
      end
      imdbRotated.images.data(:,:,:,valBatchIndex)=gather(valImages);
    end
end
