function opts = updateOptsUnc(varargin )
%% function opts = updateOptsUnc(param1,val1,param2,val2,param3,val3 ....)
% updateOptsUnc gets a set of parameters required for polar layer and
% sets fields of opts according to inputs
% 
%
% Inputs:
% atten_LR  = 0         learning rate for the moving of center of attention
%
% isNormalize = false   normalizes the movement:: not recommended : buggy
% isMaximize 
opts.atten_LR = 0;
opts.isNormalize = false;
opts.isMaximize = true;
opts = vl_argparse(opts,varargin);
end

