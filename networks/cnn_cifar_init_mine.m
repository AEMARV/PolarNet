function net = cnn_cifar_init_nin(varargin)
opts.networkType = 'simplenn' ;
opts = vl_argparse(opts, varargin) ;

% CIFAR-10 model from
% M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013.
LayerNum =18;
MAXCHANNEL = 32;
MINCHANNEL = 64;
survRate = 0.8;
ConvSizeSet = [1,3,5,7];
ConvSize = rand(1,LayerNum);
ConvSize = floor(ConvSize * numel(ConvSizeSet))+1;
ConvSize = ConvSizeSet(ConvSize);
ChannelSize = floor(rand(1,LayerNum).*(MAXCHANNEL-MINCHANNEL)) +MINCHANNEL;
ChannelSize(1)= 3;
ChannelSize(end+1) = 10;
net.layers = {} ;
b=0 ;
for i  = 2 : LayerNum +1
net.layers{end+1} = struct('type', 'conv', ...
                           'name', 'conv1', ...
                           'weights', {{0.01*randn(ConvSize(i-1),ConvSize(i-1),ChannelSize(i-1),ChannelSize(i),'single'), b*ones(1,ChannelSize(i),'single')}}, ...
                           'learningRate', [.1 2], ...
                           'stride', 1, ...
                           'pad', (ConvSize(i-1)-1)/2) ;
net.layers{end+1} = struct('type', 'relu', 'name', 'stochrelu1') ;
if i ~= LayerNum +1
net.layers{end+1} = struct('type', 'dropchannel', 'name', 'stochrelu1','rate',1- sqrt(survRate)) ;
net.layers{end+1} = struct('type', 'dropout', 'name', 'stochrelu1','rate', sqrt(survRate)) ;
end
end
net.layers{end+1} = struct('name', 'pool2', ...
                           'type', 'pool', ...
                           'method', 'max', ...
                           'pool', [32 32], ...
                           'stride', 1, ...
                           'pad', 0) ;

net.layers{end+1} = struct('type', 'softmaxloss') ;

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.learningRate = [0.005*ones(1,30) 0.1*ones(1,10) 0.02*ones(1,5) 0.001*ones(1,20)]  ;
net.meta.trainOpts.weightDecay = 0.0005 ;
net.meta.trainOpts.batchSize = 100 ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% Fill in default values
net = vl_simplenn_tidy(net) ;

% Switch to DagNN if requested
switch lower(opts.networkType)
  case 'simplenn'
    % done
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
      {'prediction','label'}, 'error') ;
  otherwise
    assert(false) ;
end
