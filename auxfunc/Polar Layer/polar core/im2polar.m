function UpSampledPolar = im2polar(M,Center,opts)
%% im2polar converts an image from cartesian to polar coordinates
%
% function Zinterp = im2polar(M,Center,opts)
% M is the image which should be single gpuArray
%--------------------------------------------------------------------------
% Center determines the center of polar coordinates, on input -1 it is the 
% center of image  Center(1) is the row and Center(2) is the column coor
%--------------------------------------------------------------------------
% Zinterp is the polar coordinates at center which is upsampled by factor
% defined in ops with size of size(M,1)*UP
%
% columns show different theta ( X axis)
% rows show different radius ( Y axis ) 
%
%
% --------attention
% opts is a struct ('type', 0('log') or 1('lin') or 2('square'),
% 'upSampleRate',UP,'filterSigma',FS , 'extrapval' , extrapval,'kernel',k OPTIONAL: rmin ,rmax)

%% parsing options
if nargin>2
    type = opts.typePolar;
    upSampleRate = opts.upSampleRate;
    upSampleSize = opts.upSampleRate * size(M,1);
    filterSigma = opts.filterSigma;
    kernel = opts.kernel;
    extrapval = opts.extrapval;
    if isfield(opts,'rmin')
        rmin = opts.rmin;
    else
        rmin = gpuArray.zeros(1);
    end
    if isfield(opts,'rmax')
        rmax = opts.rmax;
    else
        rmax = size(M,1)/2;
    end
else
    type = 1;
    upSampleRate = 2;
    upSampleSize = upSampleRate * gpuArray.size(M,1);
    filterSigma = upSampleRate/3;
    extrapval = 0;
end
X0 = Center(:,1);
Y0 = Center(:,2);

%% checking class of inputs.
 if (~isa(M,'gpuArray')) || ~isa(Center,'gpuArray')
     error('not gpuArray');
 end
%% upsampling in polar
 
 UpSampledPolar = upSamplePolar(M,X0,Y0,upSampleSize,rmin,rmax,extrapval,type);
 
 
 
end
function [GridTheta,GridR] = createGridsPolar(type,GridNum,rmax)
    %% function [GridTheta,GridR] = createGridsPolar(type,GridNum,rmax)
    % where type can be 'square' , 'lin' , 'log'
if type == 2
        rhoi = gpuArray.linspace(0,sqrt(rmax),GridNum);
        thetai = gpuArray.linspace(-pi,pi,GridNum);
        rhoi = rhoi.^2;
        [GridTheta,GridR] = meshgrid(thetai,rhoi);
   %error('not implemented yet');
elseif type == 1
        rhoi = gpuArray.linspace(0,rmax,GridNum);
        thetai = gpuArray.linspace(-pi,pi,GridNum);
        [GridTheta,GridR] = meshgrid(thetai,rhoi);
elseif type == 0
     %% plus 1 because alpha should remain positive
%    alpha = GridNum/log(rmax+1);
        % 
        rLog = gpuArray.logspace(0,log10(rmax),GridNum);
        theta = gpuArray.linspace(-pi,pi,GridNum);
        [GridTheta,GridR] = meshgrid(theta,rLog);
else
    error('unknown sampling type');    
end

end
function upSampledPolar = upSamplePolar(M,Row0,Col0,upSampleSize,rmin,rmax,extrapval,type)
%% function upSampledPolar = upSamplePolar(M,X0,Y0,upSampleSize,rmin,rmax,extrapval,type)
% creates the upsampled polar transform of image
% where M is W*H*3 and X0 and Y0 are the centers in terms of pixels from
% top and left
% rmin sets the minimum distant from center to be sampled
% ramx sets the maximum distant from center to be sampled
% extrapval is the value for out of bound sampling, nan results in random
% upSampleSize is the final output size
% type can be 'log' 'square' 'lin'

[thetai,rhoi] =createGridsPolar(type,upSampleSize,rmax);
one = gpuArray.ones(1);
CHANNELNUM = size(M,3);
BATCHSIZE = size(M,4);
SIZEM1 = gpuArray(size(M,1));
SIZEUP = gpuArray(size(thetai,1));
 % creates up sampled polar with equi spaced rings and radial lines
% upSampledPolar =gpuArray.ones(upSampleSize,upSampleSize,size(M,3),size(M,4));
 % rounds and calculates the cartesian domain equivalent of polar
 % coordinate, measures in space P0 (image original space)
 % rhoi is 1 : log(rmax+1) thats why rhoi -1
 
 PureRow = (rhoi).*sin(thetai);
 PureCol = (rhoi).*cos(thetai);
 PureRow = repmat(PureRow,1,1,CHANNELNUM,BATCHSIZE);
 PureCol = repmat(PureCol,1,1,CHANNELNUM,BATCHSIZE);
 Row0 = reshape(Row0,1,1,1,BATCHSIZE);
 Col0 = reshape(Col0,1,1,1,BATCHSIZE);
 Row0 = repmat(Row0,SIZEUP,SIZEUP,CHANNELNUM,1);
 Col0 = repmat(Col0,SIZEUP,SIZEUP,CHANNELNUM,1);
 
 RowSample = round(PureRow + Row0);
 ColSample = round(PureCol + Col0);
 %RowSample = round((rhoi-one).*sin(thetai)+Y0);
 %ColSample = round((rhoi-one).*cos(thetai)+X0);
 % sets the out of bound values to 1
 OUT = RowSample > SIZEM1 | RowSample < one | ColSample > SIZEM1 | ColSample < one;
 outBound = find(OUT);
 
 RowSample(outBound) = one;
 ColSample(outBound) = one;
 %RowSample(RowSample < one) = one;
 %ColSample(ColSample > SIZEM1) = one;
 %ColSample(ColSample < one) = one;
 % gets the polar indices for the first channel 
 % should add K* UpsampledPolar(1)* UpsampledPolar(2)
 %PixelNum = SIZEM1* SIZEM1;
 
 sub2indChannel = gpuArray.colon(1,CHANNELNUM);
 sub2indChannel = reshape(sub2indChannel,1,1,CHANNELNUM,1);
 sub2indChannel = repmat(sub2indChannel,SIZEUP,SIZEUP,1,BATCHSIZE);
 
 sub2indBatch = gpuArray.colon(1,BATCHSIZE);
 sub2indBatch = reshape(sub2indBatch,1,1,1,BATCHSIZE);
 sub2indBatch = repmat(sub2indBatch,SIZEUP,SIZEUP,CHANNELNUM,1);
 polarIndices = sub2ind([SIZEM1,SIZEM1,CHANNELNUM,BATCHSIZE], RowSample, ColSample,sub2indChannel,sub2indBatch);
 %polarIndices(outBound) = one;
 upSampledPolar = M(polarIndices);

 % replace out bounds
 if isnan(extrapval)
    
 upSampledPolar(outBound) = 255*(rand(1,numel(outBound))-0.5);
 
 
 else
 upSampledPolar(outBound)                   =  extrapval;
 
 end
end
