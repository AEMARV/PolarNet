    function  layer  = createPolarLayer(opts)
%% function  layer  = createPolarLayer(opts)
% creates a polar layer with the following fields
% ========================================================================
%
% Fields:
% ------------------------------------------------------------------------
% opts : contains options about the polar transform
% opts is a struct (
%
%   'type' :  'log' or 'lin' or 'square'
%
%   'upSampleRate',UP,
%
%   'filterSigma',FS , 
%
%   'extrapval' , extrapval,
%
%   'kernel',k     ;;  There are constraints on size of the kernel and sigma
%
%   OPTIONAL: rmin ,rmax
%
% ------------------------------------------------------------------------
% type : 'custom'  this is fixed and does not depend on input
% ------------------------------------------------------------------------
% centers : the default value is [] if it remains [] during training it
% will throw error
% one should update centers which is a 2*2*1*N matrix for every batch

%layer.opts = opts;

layer.kernel = opts.kernel;
layer.extrapval = opts.extrapval;
layer.upSampleRate = opts.upSampleRate;
layer.DownSampleRate = opts.DownSampleRate;
layer.typePolar = opts.type;
layer.filterSigma = opts.filterSigma;
layer.randomRotate = opts.randomRotate;
layer.rotatePix = 0;
layer.convFreq = opts.convFreq;
if isfield(opts,'rmin')
    layer.rmin = opts.rmin;
    layer.rmax = opts.rmax;
end
layer.type  = 'custom';
layer.forward = @pol_transform_wrapper_forward;
layer.backward = @pol_transform_wrapper_backward;
layer.centers = [];

end
function resip1 =  pol_transform_wrapper_forward(layer,resi,resip1)

    
    centers = layer.centers;
    if isempty(centers)
        error('centers are empty matrix');
    end
    if ndims(centers) == 4
    centers = centers(1,:,1,:);
    centers = squeeze(centers)';
    end
    resip1.x = pol_transform(resi.x,centers,layer);
    if layer.randomRotate
    shiftAmount = rand(1);
    shiftAmount = floor(shiftAmount * size(resi.x,2));
    layer.rotatePix = shiftAmount;
    resip1.x = shiftAll(resip1.x,layer.rotatePix);
    end
    
    
    if layer.convFreq
        resip1.x = FreqConv(resip1.x);
    end
end
function resi = pol_transform_wrapper_backward(layer,resi,resip1)
%    opts = layer.opts;
    
    centers = layer.centers;
    if isempty(centers)
        error('centers are empty matrix');
    end
    dzdpol = resip1.dzdx;
    if layer.convFreq
        dzdpol = iFreqConv(dzdpol);
    end
    if layer.randomRotate
       dzdpol = shiftAll(dzdpol,-layer.rotatePix); 
    end
    
    [dzdrow,dzdcol] = calcGradCenter(dzdpol,resi.x,centers,layer);
    resi.dzdrow = dzdrow;
    resi.dzdcol = dzdcol;
end
function shifted = shiftAll(x,shiftAmount)
colNum = size(x,2);
shifted = x;
ind = 0: colNum-1; % 0 15
indNew = mod((ind + shiftAmount),colNum) +1 ;% SA : 15+SA -> SA : 15 : 0: SA -1
shifted(:,ind+1,:,:) = x(:,indNew,:,:);
end
function FreqIm = FreqConv(ims,isRadius)
% converts the ims images with W*H*C*B dimensions into frequency domain in
% each row (Radius); 
if nargin <2
    isRadius = true;
end
if isRadius
ims = permute( ims,[2,1,3,4]);
% NOW ROWS ARE DIFFERENT THETAS
end
SIZE = size(ims);
RowNum = SIZE(1);
ColNum = SIZE(2);
ChanNum = SIZE(3);
BatchNum = SIZE(4);

ims = reshape(ims,[RowNum , ColNum * ChanNum * BatchNum]);
FreqIm = dct(ims);
FreqIm = reshape(FreqIm,[RowNum,ColNum,ChanNum,BatchNum]);

if isRadius
   FreqIm = permute( FreqIm,[2,1,3,4]);
end



end
function Im = iFreqConv(FreqIm,isRadius)
if nargin <2
    isRadius = true;
end
if isRadius
FreqIm = permute( FreqIm,[2,1,3,4]);
% NOW ROWS ARE DIFFERENT THETAS
end
SIZE = size(FreqIm);
RowNum = SIZE(1);
ColNum = SIZE(2);
ChanNum = SIZE(3);
BatchNum = SIZE(4);

FreqIm = reshape(FreqIm,[RowNum , ColNum * ChanNum * BatchNum]);
Im = idct(FreqIm);
Im = reshape(Im,[RowNum,ColNum,ChanNum,BatchNum]);

if isRadius
   Im = permute( Im,[2,1,3,4]);
end
end