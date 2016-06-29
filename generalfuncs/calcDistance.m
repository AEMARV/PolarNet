function [ dist ] = calcDistance( vectorIn)
%CALCDISTANCE Summary of this function goes here
%   Detailed explanation goes here
    dist = sum(vectorIn(:).^2);

end

