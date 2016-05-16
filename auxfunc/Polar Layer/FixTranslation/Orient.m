function [rowShift,colShift] = Orient( res,beginLayer,EndLayer )
%ORIENT Summary of this function goes here
%   Detailed explanation goes here
SIZEBASE = size(res(beignLayer).x);
Fields = fieldnames(res);
dzdrowFieldInd = find(strcmp(Fields,'dzdrow'));
dzdcolFieldInd = find(strcmp(Fields,'dzdcol'));
resCell = struct2cell(res);
% resCell Field * 1 * ALL_Layer Cell
dzdRowCell = resCell(dzdrowFieldInd,:,beginLayer:EndLayer);
dzdColCell = resCell(dzdColFieldInd,:,beginLayer:EndLayer);
dzdRowCell = reshape(dzdrowCell,[1,numel(dzdRowCell)]);
dzdColCell = reshape(dzdColCell,[1,numel(dzdColCell)]);
%dzdCol/RowCell   1 * Layer Cell
dzdRowArray = cell2mat(dzdRowCell);
dzdColArray = cell2mat(dzdColCell);
% matrix of size B*LayerNum
SizeMat = createSizeVector(res,beginLayer,EndLayer,1);
SizeRowMat = SizeMat(:,1);
SizeColMat = SizeMat(:,2);

MapRow = createRateMap(SizeRowMat,BatchSize);
MapCol = createRateMap(SizeColMat,BatchSize);
rowShift = dzdRowArray.*MapRow;
colShift = dzdColArray.*MapCol;
rowShift = sum(rowShift,2);
colShift = sum(colShift,2);
end
function calcAbsShift(sizeBase,sizeCurrent,resi)
end
function SIZE_VEC = createSizeVector(res,beginLayer,EndLayer,fieldIndex)
% creates an Array contianing the size of the dimension
% specified(rows[1]/columns[2]).
% outputs the size where the rows shows the layers.fieldName size
resCell = struct2cell(res);
xCells = resCell(fieldIndex,:,beginLayer:EndLayer);
xCells  = xCells(:);
SIZE_VEC = cellfun(@size,xCells,'UniformOutput');
end
function MAP = createRateMap(dimMat,BatchSize)
% creates learning rates for each layer
% the output is BATCHSIZE * LAYERS
relDim = dimMat./ dimMat(1);
MAP = repmat(relDim,[BatchSize,1]);

end
