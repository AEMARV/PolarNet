function out = runALL()
%% 
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
useUncertainty = false;
useGmm = false;
upSampleRate =double(2);
DownSampleRate = double(2);
filterSigma = single(2/3);
interval = 0;
extrapvalue = 0;
uncOpts = [];
%% uncertainty option
atten_LR = 0.05;
isNormalize = false;
isMaximize = true; % if set maximizes the certainty when moving
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
    ,'uncOpts',uncOpts);

cnn_cifar('train',struct('gpus',1),'expDir','./results'...
    ,'usePolar',polarOpts.usePolar...
    ,'polarOpts',polarOpts);
end