function [centers, imdb,batch ] = getCentersImdb( imdb,batch,flip )
%% function [centers, imdb,batch ] = getCentersImdb( imdb,batch,flip )
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
    
    centers = gpuArray(imdb.images.centerHist(:,:,:,batch));
    if flip
    centers(1,2,:,:) = 1- centers(1,2,:,:);
    end

end

