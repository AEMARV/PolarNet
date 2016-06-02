function  isSafe = StructVerifier(StructIn,type )
%function  isSafe = StructVerifier(StructIn,type )
%   % type is {dataparam,[not implemented]};
switch lower(type)
    case 'dataparam'
        
        isSafe = checkDataParam(StructIn);
        
    otherwise
        error('unknown structure')
end


end
function isSafe = checkDataParam(inStruct)
isSafe = false;
FNUM = 5;
assert(hasAllFields(inStruct,'row0','col0','rmin','rmax','theta0'));

Fnames = fieldnames(inStruct);

for i = 1 : FNUM
    varI = inStruct.(Fnames{i});
    if strcmp(Fnames{i},'rmin')
        varMax = inStruct.rmax;
        DiffMaxMin = varMax - varI;
        %assert(gather(all(DiffMaxMin>0)),'rmax must be beq to rmin');
        %assert(gather(all(varI>=0)),'rmin cannot be less than 0');
    end
    if i ~= 1
        assert(gather(all(temp == size(varI))),'fields in the DataParam struct has different sizes');
        temp = size(varI);
    else
        temp = size(varI);
    end
    assert(gather(all(size(varI,2)==1)),'%s field in the struct has invalid size',Fnames{i});
end
isSafe = true;
end
function isSafe = hasAllFields(inStruct,varargin)
for i = 1 : size(varargin)
    if ~isfield(inStruct,varargin{i})
        warning('field %s does not exist in the current struct',varargin{i});
        isSafe = false;
        return
    end
end
isSafe = true;
end
