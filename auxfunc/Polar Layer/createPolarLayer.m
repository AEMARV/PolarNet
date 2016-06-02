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
layer.useLocator = true;
layer.LocatorNet = cnn_cifar_init_loc();
layer.LocatorRes = [];
layer.epoch = 0;
layer.useCenters = false;
if isfield(opts,'rmin')
    layer.rmin = opts.rmin;
    layer.rmax = opts.rmax;
end
layer.type  = 'custom';
layer.forward = @pol_transform_wrapper_forward;
layer.backward = @pol_transform_wrapper_backward;
layer.DataParam = [];

end
function resip1 =  pol_transform_wrapper_forward(layer,resi,resip1)
    LOCATE = layer.useLocator && layer.epoch > 5;
    if LOCATE
        % resip1.aux is the forward pass result
        [resip1.DataParam,resip1.aux] = locate(layer.LocatorNet,resi.x,resip1.aux,resi.DataParam);
        StructVerifier(resip1.DataParam,'dataparam');
        dataParam = resip1.DataParam;
        if layer.useCenters
            dataParam.row0 = resi.DataParam.row0;
            dataParam.col0 = resi.DataParam.col0;
        end
    else
        [resip1.DataParam,resip1.aux] = locate(layer.LocatorNet,resi.x,resip1.aux,resi.DataParam);
        dataParam = resi.DataParam;
        StructVerifier(dataParam,'dataparam');
    end
    resip1.x = pol_transform(resi.x,dataParam,layer);
   
end
function resi = pol_transform_wrapper_backward(layer,resi,resip1)
%    opts = layer.opts;
    LOCATE = layer.useLocator && layer.epoch > 5 ;
    if LOCATE
    dataParam = resip1.DataParam;
    else
        dataParam = resi.DataParam;
    end
  
    dzdpol = resip1.dzdx;
    [resi.dzdx,resi.dzdDataParam] = pol_transform(resi.x,dataParam,dzdpol);
    resi.dzdDataParam = regrminrmax(resi.dzdDataParam,dataParam);
    if LOCATE
    [resi.aux,resip1.aux] = backPropLocator(layer.LocatorNet,resi.x,resip1.aux,resi.dzdDataParam,   layer);
    else
          resi.aux = layer.LocatorNet;
          
%         dzdDataParam = (-dataParam2arr(   createBaseDP(resi.dzdDataParam))+dataParam2arr(resip1.DataParam));
%         dzdDataParam = arr2dataparam(sign(dzdDataParam),resi.dzdDataParam);
%         [resi.aux,resip1.aux] = backPropLocator(layer.LocatorNet,resi.x,resip1.aux,dzdDataParam,   layer);
    end
end
function dzdDataParam = regrminrmax(dzdDataParam,DataParam)
    IndRmin = find(DataParam.rmin < 0);
    dzdDataParam.rmin(IndRmin) = dzdDataParam.rmin(IndRmin) -1;
    IndRminRmax = find(DataParam.rmin > DataParam.rmax);
    dzdDataParam.rmin(IndRminRmax) = dzdDataParam.rmin(IndRminRmax) +1;
    dzdDataParam.rmax(IndRminRmax) = dzdDataParam.rmax(IndRminRmax) -1;
end
function shifted = shiftAll(x,shiftAmount,rowType,colType)
% shifts x where x is M*N*C*B according to shiftAmount where
% shiftAmount is a 2 * B matrix: first row is row shift and second is col shift
% rowType specified the 
shifted = shiftFirstDim(x,shiftAmount(1,:)',rowType);
shifted = permute(shiftFirstDim(permute(shifted,[2,1,3,4]),shiftAmount(2,:)',colType),[2,1,3,4]);
end
function shifted = shiftFirstDim(x,shiftAmount,type)
SIZE = size(x);
MAXPAD = gather(max(abs(shiftAmount(:))));
if MAXPAD ==0
    shifted = x;
    return;
end
shifted = padarray(x,[MAXPAD,0,0,0],type,'both');
rowInd = 1+MAXPAD:SIZE(1)+MAXPAD;
[rowSub,colSub,chSub,Bsub] = ndgrid(rowInd,1:SIZE(2),1:SIZE(3),1:SIZE(4));
shiftAmount = reshape(shiftAmount,1,1,1,SIZE(4));
%[~,~,~,ShiftRow] = ndgrid(rowInd,1:SIZE(2),1:SIZE(3),shiftAmount);
%newRowSub = rowSub + ShiftRow;
newRowSub = bsxfun(@plus,rowSub,shiftAmount);
Ind = sub2ind(size(shifted),newRowSub,colSub,chSub,Bsub);
shifted = shifted(Ind);
end
function [dataParam,res] = locate(net_gpu,im,res,dataParam)
BATCH_SIZE = size(im,4);
res = vl_simplenn(net_gpu,im,[],[],res,'CuDNN',true);
dataParamArr =res(end).x;
dataParamArr = squeeze(permute(dataParamArr,[4,3,2,1]));
Fnames = fieldnames(dataParam);
for i = 1 : numel(Fnames)
    dataParam.(Fnames{i}) = dataParamArr(:,i);
end
end
function [net,res] = backPropLocator(net,im,res,dzdDataParam,layer)
    
    dzd_DP_arr = dataParam2arr(dzdDataParam);
    dzd_DP_arr = ipermute(dzd_DP_arr,[4,3,2,1]);
    if ~isempty(res)
    res(end).dzdx = dzd_DP_arr;
    end
    res = vl_simplenn(net,im,[],dzd_DP_arr,res,'CuDNN',true,'SkipForward',true);
    net = Learn(net,res,layer);
end
function DParr = dataParam2arr(DP)
    Fnames = fieldnames(DP);
    BATCHSIZE = size(DP.(Fnames{1}),1);
    paramNum = numel(Fnames);
    DParr =  gpuArray.zeros([BATCHSIZE,paramNum],'single');
    for i = 1 : paramNum
        DParr(:,i) = DP.(Fnames{i});
    end
end
function DP = arr2dataparam(DParr,DP)
    Fnames = fieldnames(DP);
    BATCHSIZE = size(DP.(Fnames{1}),1);
    paramNum = numel(Fnames);
    DParr =  gpuArray.zeros([BATCHSIZE,paramNum],'single');
    for i = 1 : paramNum
       DP.(Fnames{i}) = DParr(:,i);
    end
end
function DP = createBaseDP(DP)
Fnames = fieldnames(DP);
    BATCHSIZE = size(DP.(Fnames{1}),1);
    paramNum = numel(Fnames);
    for i = 1 : paramNum
        switch Fnames{i}
            case 'rmin'
                arr = gpuArray.zeros(BATCHSIZE,1);
            case 'rmax'
                arr = gpuArray.ones(BATCHSIZE,1);
            case 'theta0'
                arr = gpuArray.zeros(BATCHSIZE,1);
            case 'row0'
                arr = gpuArray.zeros(BATCHSIZE,1);
            case 'col0'
                arr = gpuArray.zeros(BATCHSIZE,1);
            otherwise
                error('invalid field name')
        end
       DP.(Fnames{i}) = arr;
    end
    
end

function net = Learn(net,res,layer)
trainOpts = net.meta.trainOpts;
batchSize = trainOpts.batchSize;
LR = trainOpts.learningRate(layer.epoch);
for l=numel(net.layers):-1:1
  for j=1:numel(res(l).dzdw)

    % accumualte gradients from multiple labs (GPUs) if needed
   

    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = ...
        (1 - thisLR) * net.layers{l}.weights{j} + ...
        (thisLR/batchSize) * res(l).dzdw{j} ;
    else
      % standard gradient training
      thisDecay = trainOpts.weightDecay * 1 ;
      thisLR = LR * net.layers{l}.learningRate(j) ;
      Grad = - thisDecay * net.layers{l}.weights{j} ...
        - (1 / batchSize) * res(l).dzdw{j} ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} + ...
        thisLR * Grad ;
    end
  end
end
end