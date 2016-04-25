function out = runALL(varargin)
%% 
% creating videos
skip = true;

if skip
    runCreateVideo();
    return
end

% polar options
typePolar = 'log';
switch typePolar
    case 'log'
        typePolar = 0;
    case 'linear'
        typePolar = 1;
    case 'square'
        typePolar = 2;
    otherwise
        warning('setting the polar type to default: linear \n')
        typePolar = 1;
end
contnu = false;% continue parameter
usePolar = true;
useUncertainty = true;
useGmm = false;
upSampleRate =double(2);
DownSampleRate = double(2);
filterSigma = single(2/3);
interval = 0;
extrapvalue = 0;
uncOpts = [];
%% uncertainty option
atten_LR = 0.1;
isNormalize = false;
isMaximize = false; % if set maximizes the certainty when moving
if useUncertainty
uncOpts = updateOptsUnc('atten_LR',atten_LR...
    ,'isNormalize',isNormalize...
    ,'isMaximize',isMaximize);
end
polarOpts = updateOptsPolar(...
     'continue',contnu...
    ,'type',typePolar...
    ,'usePolar',usePolar...
    ,'useUncertainty',useUncertainty...
    ,'useGmm',useGmm...
    ,'upSampleRate',upSampleRate...
    ,'DownSampleRate',DownSampleRate...
    ,'filterSigma',filterSigma...
    ,'interval',interval...
    ,'extrapval',extrapvalue...
    ,'uncOpts',uncOpts...
    ,'randomRotate',true);

cnn_cifar('train',struct('gpus',1),'expDir','./results'...
    ,'usePolar',polarOpts.usePolar...
    ,'polarOpts',polarOpts);
runCreateVideo();
end
function runCreateVideo(varargin)
opts.dataBaseName = 'cifar';
opts.imdbPath = './results';
opts.netPath = opts.imdbPath;
opts.imagePath = fullfile(vl_rootnn(),'data','cifar','cifar-10-batches-mat');
opts.movieOutPathBase = fullfile(pwd,'evolveVids');
opts.videoRes = 32;
opts.numberOfVideos = 10;
opts.BatchSize =1024;
opts =vl_argparse(opts,varargin);
createVideos('dataBaseName',opts.dataBaseName...
,'imdbPath',opts.imdbPath...
,'netPath',opts.netPath...
,'imagePath',opts.imagePath...
,'movieOutPathBase',opts.movieOutPathBase...
,'videoRes',opts.videoRes...
,'numberOfVideos',opts.numberOfVideos...
,'BatchSize',opts.BatchSize);
end