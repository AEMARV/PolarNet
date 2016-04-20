function opts = updateOptsPolar(varargin)
%% function opts = updateOptsPolar(opts,type,upSampleRate,DownSampleRate,filterSigma,interval,extrapval)
% updateOptsPolar gets a set of parameters required for polar layer and
% sets fields of opts according to inputs
% if the number of input arguments is 1, it will set the values to the
% default values.
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


    opts.
    opts.type = 0;
    opts.usePolar = true;
    opts.useUncertainty = false;
    opts.useGmm = false;
    opts.upSampleRate = double(2);
    opts.DownSampleRate = double(2);
    opts.filterSigma = single(2/3);
    opts.interval = 6;
    opts.kernel = single(fspecial('gaussian',ceil(double(2*opts.filterSigma *3)),double(opts.filterSigma)));
    opts.extrapval = single(0);
    opts = vl_argparse(opts,varargin);
    opts.kernel = single(fspecial('gaussian',ceil(double(2*opts.filterSigma *3)),double(opts.filterSigma)));
    opts.extrapval = single(opts.extrapval);

end

