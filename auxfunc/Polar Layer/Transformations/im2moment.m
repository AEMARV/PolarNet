function [ imageMoment ] = im2moment( image,degree)
% function [ imageMoment ] = im2moment( image,degree)
% Calculates the cumulative moment of order n along the row axis,
% imageMoment(row,column,channel) = row^degree*image(row,col,channel) +
% imageMoment(row-1,col,channel) each row is the moment of image(1:row,:,:)
%
% - input:
%           "image" is a matrix of size WxHx(C*B)
%           "degree" is the degree of moment which the image needs to
%               converted to.
% - output:
%           "imageMoment" is a matrix the same size of the image
    if(nargin==2)
        row = size(image,1);
        col = size(image,2);
        channels = size(image,3);
        [~,rowIndex] = meshgrid(1:col,1:row,1:channels);
        weights = rowIndex.^degree;
        imageMoment = image.*weights;
        imageMoment = cumsum(imageMoment);

    end
end

