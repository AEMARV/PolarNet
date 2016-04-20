function output_args  = pol_transform( input_args,centers,opts )
% function output_args  = pol_transform( input_args,centers,opts )
%
% centers are normalized between 0:1
% centers are N*2. first column is the row and second is the column coor
%
% opts is a struct ('typePolar','log' or 'lin' or 'square',
% 'upSampleRate',UP,'filterSigma',FS , 'extrapval' , extrapval,'kernel',k OPTIONAL: rmin ,rmax)
if nargin>2
    type = opts.typePolar;
    upSampleRate = opts.upSampleRate;
    upSampleSize = opts.upSampleRate * size(input_args,1);
    filterSigma = opts.filterSigma;
    kernel = opts.kernel;
    extrapval = opts.extrapval;
    if isfield(opts,'rmin')
        rmin = opts.rmin;
    else
        rmin = 0;
    end
    if isfield(opts,'rmax')
        rmax = opts.rmax;
    else
        rmax = size(input_args,1)/2;
    end
else
    type = 1;
    upSampleRate = 2;
    upSampleSize = upSampleRate * size(M,1);
    filterSigma = upSampleRate/3;
    extrapval = 0;
end



BATCH_SIZE = size(centers,1);
CHANNEL_SIZE = size(input_args,3);

SIZE1 = size(input_args,1);
SIZE2 = size(input_args,2);
centers(:,1) = SIZE1 * centers(:,1);
centers(:,2) = SIZE2 * centers(:,2);

padAmount = double(gather(ceil((size(opts.kernel,1) - opts.upSampleRate)/2)));



%parfor i = 1: size(input_args,4)
       
          UpSampledPolar = im2polar(input_args,centers,opts);
          %fprintf('%f % processed \n',i/size(input_args,4));
          %fprintf('\n');
      
%end
SIZEUP = SIZE1 * opts.upSampleRate;
one_channelUpSampled = reshape(UpSampledPolar,SIZEUP,SIZEUP,1,[]);
UpSampledPolarPaded = padarray(one_channelUpSampled,[0,2*padAmount,0,0],'circular','post');
UpSampledPolarPaded = padarray(UpSampledPolarPaded,[2*padAmount,0,0,0],'post');
% 
SmoothPolar = vl_nnconv(UpSampledPolarPaded,kernel,[],'stride',opts.DownSampleRate,'CuDNN');
%SmoothPolar = SmoothPolar(:,1:SIZE2,:,:);
if(SIZE1 ~= size(SmoothPolar,1))
    warning('the size of the input file and the output polar is not the same');
    warning('inputSize: %d by %d \n Polar Size : %d by %d', SIZE1,SIZE2, size(SmoothPolar,1),size(SmoothPolar,2))
end
output_args = reshape(SmoothPolar,size(SmoothPolar,1),size(SmoothPolar,2),CHANNEL_SIZE,[]);
    
end

