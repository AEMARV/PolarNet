function [ Cells ] = setCellNum( Cells,num )
%SETCELLNUM Summary of this function goes here
%   Detailed explanation goes here
for i = 1 : numel(Cells)
    if iscell(Cells{i})
        Cells{i} = setCellNum(Cells{i},num);
    else
        Cells{i} = num .* ones(size(Cells{i}),'like',Cells{i});
    end
end

