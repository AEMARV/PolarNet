function [rowShift,colShift] = calc_abs_shift( res,beginLayer,EndLayer )
%% finds the amount of shift in the direction of increasing z
% function [rowShift,colShift] = calc_abs_shift( res,beginLayer,EndLayer )
% 
% beginLayer and EndLayer is the index of starting and ending layers that use the same
% coordinate system. in the case of lenet with polar layer is 2:14 or 13
SIZEBASE = size(res(beginLayer).x);
Fields = fieldnames(res);
dzdrowFieldInd = find(strcmp(Fields,'dzdrow'));
dzdcolFieldInd = find(strcmp(Fields,'dzdcol'));
resCell = struct2cell(res);
% resCell Field * 1 * ALL_Layer Cell
dzdRowCell = resCell(dzdrowFieldInd,:,beginLayer:EndLayer);
dzdColCell = resCell(dzdcolFieldInd,:,beginLayer:EndLayer);
dzdRowCell = reshape(dzdRowCell,[1,numel(dzdRowCell)]);
dzdColCell = reshape(dzdColCell,[1,numel(dzdColCell)]);
%dzdCol/RowCell   1 * Layer Cell
%dzdRowArray = cell2mat(dzdRowCell);
dzdRowArray = vl_nnconcat(dzdRowCell,2);
dzdColArray = vl_nnconcat(dzdColCell,2);
% matrix of size B*LayerNum
SizeMat = createSizeVector(res,beginLayer,EndLayer,1);
SizeRowMat = SizeMat(:,1);
SizeColMat = SizeMat(:,2);
BatchSize = SizeMat(1,4);
rowShift = calcShift(SizeRowMat,dzdRowArray);
colShift = calcShift(SizeColMat,dzdColArray);

end
function SIZE_VEC = createSizeVector(res,beginLayer,EndLayer,fieldIndex)
% creates an Array contianing the size of the dimension
% specified(rows[1]/columns[2]).
% outputs the size where the rows shows the layers.fieldName size
resCell = struct2cell(res);
xCells = resCell(fieldIndex,:,beginLayer:EndLayer);
xCells  = xCells(:);
SIZE_VEC = cellfun(@size,xCells,'UniformOutput',false);
SIZE_VEC = cell2mat(SIZE_VEC);
end
function totalShift = calcShift(dimMat,dzdshift)
% creates learning rates for each layer
% the output is BATCHSIZE * LAYERS
% you can change the optimization policy here
% dzdshift is B * layer
[dzdshiftAv,dimMatComp] = averageShift(dimMat,dzdshift);
% dimMatComp = 1* uniquesSizes
% dzdshiftAv = B* uniqueSizes
%% threshold the derivatives for stability 
stridePerShift = dimMatComp./ dimMatComp(1);
stridePerShift = 1./stridePerShift;
shiftCostPerPix = bsxfun(@rdivide,dzdshiftAv,stridePerShift');
% shiftCostPerPix shows the change of loss per pixel in each layer size and
% is B*uniquSizes
% Mask = abs(shiftCostPerPix) < abs(shiftCostPerPix(1,1));
isConsiderable = bsxfun(@minus,abs(shiftCostPerPix),abs(shiftCostPerPix(:,1)));
isConsiderable = isConsiderable >= 0;
thresholdedShift = sign(dzdshiftAv) .* isConsiderable;
%% end of thresholding
totalShift = bsxfun(@times,stridePerShift',thresholdedShift);
totalShift = sum(totalShift,2);

end
function [avShift,compDim] = averageShift(dimMat,dzdshift)
% averages the derivative in layers with the same dimension
% dzdshift is B*layer
Groups = findGroups(dimMat);
CumDer = cumsum(dzdshift,2);
finalSum = CumDer(:,Groups(:,2));
MinusComp = finalSum(:,1:end -1);
finalSum(:,2:end) = finalSum(:,2:end) - MinusComp;
GroupRep = Groups(:,2) - Groups(:,1) +1 ;
avShift = bsxfun(@rdivide,finalSum,GroupRep');
compDim = Groups(:,3);
end
function Groups = findGroups(dimMat)
% finds the groups of consequetive layers with the same size as input
% outputs Groups with each row represents a group. each row is 
%  Groups [startInd1, endInd1 , value ; .... ; startInd_N endInd_N]
assert(issorted(dimMat) | issorted(reverse(dimMat)));
[Values,startIndices,~]= unique(dimMat,'stable');
endIndices = startIndices -1;
endIndices = endIndices(2:end);
endIndices = [endIndices;numel(dimMat)];
assert(all(size(startIndices) == size(endIndices)) & all(size(startIndices,2) == 1));
Groups = [startIndices,endIndices,Values];
end
function a = dummy(x)
a = gather(x);
end