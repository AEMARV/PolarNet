function [imdbRotated] = ImdbRotation(imdb,valsetNumber,batchsize,thetaLimit)
%% 
%% function imdbRotated = ImdbRotation(imdb,valsetNumber,batchsize,thetaLimit)
% valsetNumber is the set index for validation set
% batchsize is the batch size in order to process rotation with gpu
% Theta Limit is to set a limit for rotation. max 360 and min is 0
    valIndex = find(imdb.images.set==valsetNumber);
    images = imdb.images.data;
    imdbRotated = imdb;
    for batchNumber = 1:size(valIndex,2)/batchsize
      batchSIndex = 1+(batchNumber-1)*batchsize;batchEIndex = batchNumber*batchsize;
      valBatchIndex = valIndex(batchSIndex:batchEIndex);
      valImages = images(:,:,:,valBatchIndex);
      theta = randi(thetaLimit,1,batchsize);
      for index = 1:batchsize
          valImages(:,:,:,index)= imageRotate(valImages(:,:,:,index),theta(index),'mirror');
      end
      zeroIndex = find(valImages==0);
      valImages(zeroIndex)= 127*(rand(size(zeroIndex,1),1)-.5);
      imdbRotated.images.data(:,:,:,valBatchIndex)=valImages;
    end
end
