function opts = updateOptsPolar(varargin)
%% function opts = updateOptsPolar(param1,val1,param2,val2,param3,val3 ....)
% updateOptsPolar gets a set of parameters required for polar layer and
% sets fields of opts according to inputs
% 
%
% Inputs:
% opts : struct of options for the network
%
% type : shows the type of polar transform
%       0 : log polar
%       1 : linear Polar
%       2 : Square Polar0
%
% filterSigma : the std of the gaussian for downsampling,
%
% interval : the period of updating the centers in learning. currently not
% implemented
%
% extrapval : the value for padding the image while getting polar values
%       nan: it will padd the image with uniform random numbers
%
%
%
% ----------------- note -------------------------------------------------
% the values to be set and the default is the following
% not that the kernel is calculated seperately with filter sigma
% 
% opts.kernel = single(fspecial('gaussian',ceil(double(2*opts.filterSigma *3)),double(opts.filterSigma)));
%
%
%
%
%    opts.type = 0;
%    
%    opts.usePolar = true;
%    
%    opts.useUncertainty = false;
%    
%    opts.useGmm = false;
%    
%    opts.upSampleRate = double(2);
%    
%    opts.DownSampleRate = double(2);
%    
%    opts.filterSigma = single(2/3);
%    
%    opts.interval = 6;
%    
%    opts.kernel = single(fspecial('gaussian',ceil(double(2*opts.filterSigma *3)),double(opts.filterSigma)));
%    
%    opts.extrapval = single(0);
%    
%    opts.convFreq = false;
%    
%    opts.uncOpts = [];   %% this option is a struct, which you can
%    produce with updateOptsUnc
%    opts.randomRotate = false  randomly rotates the input images.


    opts.continue = false;
    opts.type = 0;
    opts.usePolar = true;
    opts.useUncertainty = false;
    opts.useGmm = false;
    opts.upSampleRate = double(2);
    opts.DownSampleRate = double(2);
    opts.filterSigma = single(2/3);
    opts.interval = 6;
    opts.randomRotate = false;
    opts.kernel = single(fspecial('gaussian',ceil(double(2*opts.filterSigma *3)),double(opts.filterSigma)));
    opts.extrapval = single(0);
    opts.uncOpts = updateOptsUnc;
    opts.uncOpts = [];
    opts.convFreq = false;
    opts = vl_argparse(opts,varargin);
    opts.kernel = single(fspecial('gaussian',ceil(double(2*opts.filterSigma *3)),double(opts.filterSigma)));
    opts.extrapval = single(opts.extrapval);

end

