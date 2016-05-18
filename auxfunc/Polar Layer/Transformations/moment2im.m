function [ image ] = moment2im( imageMoment, degree )
%function [ image ] = moment2im( imageMoment, degree )
% calculates inverse of cumulative moment of an image along the rows
% - 3 dimentional imageMoment with size WxHx[CxB] 
% - degree is the moment degree
% - return the image with same size as imageMoment
        row = size(imageMoment,1);
        col = size(imageMoment,2);
        channels = size(imageMoment,3);
        [~,rowIndex] = meshgrid(1:col,1:row,1:channels);
        weights = rowIndex.^(-degree);
        imageMomentShift = padarray(imageMoment,[1 0 0],'pre');
        image = imageMoment - imageMomentShift(1:end-1,:,:);
        image = image.*weights;
end

