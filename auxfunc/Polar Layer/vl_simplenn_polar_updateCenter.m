function [net,imdb,res_c] = vl_simplenn_polar_updateCenter(net,evalMode,im ,isFliped ,imdb,batch,opts,s,res_c)
   %% function [net,imdb,res_unc] = vl_simplenn_polar_train(net,im ,...
   % ... atten_LR,isFliped ,imdb,batch,isMaximize,res_c,opts,s)
   % this function gets the net with polar layer and uploads centers 
   %  into the polar layer) 
   % 
   % the polar layer) and updates the centers associated with the batch.
   % the function first mounts a certainty layer and t maximizes the
   % certainty with dzdrow and dzdcol if isMaximize is true.
   % 
   % isFliped shoud be true if the images are fliped
   % 
   % =====================================================================
   % INPUTS:
   % net : network with the polar layer and centers mounted
   % 
   % atten_LR : it is the step size for changing the center of attention
   % note that the center change is bounded by [0,1] by one pixel by default.
   % you can moddify atten_LR to change the bound
   % 
   % isFliped : flag to show that the images are fliped or not. you can
   % output this information in the getbatch function.
   %
   % imdb :
   %
   % batch : 
   % isMaximize : if true it maximizes the certainty
   % res_c : res struct which is passed here for performance issues
   % ======================================================================
   % OUTPUTS:
   % net: network with the updated centers mounted in the first layer
   %
   % imdb: updated imdb with new centers
   %
   % res_unc : result structure containing the forwardpass and backward
   % pass
   % 
   % opts  : same struct in the training function
   % s is the batch number
   cartResIndex = 1;
   if ~opts.polarOpts.useUncertainty && ~isempty(res_c)
        [res_c(1).DataParam,imdb,~] = getDataParamImdb(imdb,batch,isFliped);     
        return
   end
   atten_LR = opts.polarOpts.uncOpts.atten_LR;
   isNormalize = opts.polarOpts.uncOpts.isNormalize;
   isMaximize = opts.polarOpts.uncOpts.isMaximize;
   
   
   if ~isempty (res_c)
   [res_c(cartResIndex).DataParam,imdb] = getDataParamImdb(imdb,batch,isFliped);
   DataParam = res_c(cartResIndex).DataParam;
   else
       [DataParam,imdb] = getDataParamImdb(imdb,batch,isFliped);
   end
   if atten_LR == 0
       
       return;
   end
   unc_net = net;
   if strcmp(evalMode,'test')
   [unc_net.layers{end},~,~] = createUncertainLayer();
   end
   unc_net =vl_simplenn_tidy(unc_net);
   if isMaximize
       dzdyunc = gpuArray(1);
   else
       dzdyunc = gpuArray(-1);
   end
   res_c = vl_simplenn(unc_net, im,DataParam, dzdyunc, res_c, ...
       'accumulate', s ~= 1, ...
       'mode', evalMode, ...
       'conserveMemory', false, ...
       'backPropDepth', opts.backPropDepth, ...
       'sync', opts.sync, ...
       'cudnn', opts.cudnn) ;
   dzdDataParam = res_c(cartResIndex).dzdDataParam;
   DataParamUpd = stepStruct(res_c(cartResIndex).DataParam,dzdDataParam,atten_LR);
  imdb = setDataParamImdb(imdb,DataParamUpd,batch,isFliped);
  res_c(cartResIndex).DataParam = DataParamUpd;
        
end

function stout = multStruct(st,multip)
Fnames = fieldNames(st1);
cell_st1 = struct2cell(st1);
stMat = cell2mat(cell_st1);
stMat = stMat.*multip;
cell_st1 = mat2cell(stMat);
end
    