
function [ net ] = cnn_cifar_loc( varargin )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
opts.fcnum = 4;
opts.classnum = 5;
opts = vl_argparse(opts,varargin);
net = cnn_cifar_init('networktype','dagnn'...
    ,'usePolar',false...
    ,'classnum',opts.classnum ...
    ,'fcnum',opts.fcnum);
net.addlayer

end

