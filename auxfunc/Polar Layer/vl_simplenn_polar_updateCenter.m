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
   if ~opts.polarOpts.useUncertainty
        [net.layers{1}.centers,imdb,batch] = getCentersImdb(imdb,batch,isFliped);     
        return
   end
   atten_LR = opts.polarOpts.uncOpts.atten_LR;
   isNormalize = opts.polarOpts.uncOpts.isNormalize;
   isMaximize = opts.polarOpts.uncOpts.isMaximize;
   
   
   
   [net.layers{1}.centers,imdb,batch] = getCentersImdb(imdb,batch,isFliped);
   if atten_LR == 0
       
       return;
   end
   unc_net = net;
   CentHist = net.layers{1}.centers;
   [unc_net.layers{end},forwardHandle,backwardHandle] = createUncertainLayer();
   unc_net =vl_simplenn_tidy(unc_net);
   if isMaximize
       dzdyunc = gpuArray(1);
   else
       dzdyunc = gpuArray(-1);
   end
   res_c = vl_simplenn(unc_net, im, dzdyunc, res_c, ...
       'accumulate', s ~= 1, ...
       'mode', evalMode, ...
       'conserveMemory', opts.conserveMemory, ...
       'backPropDepth', opts.backPropDepth, ...
       'sync', opts.sync, ...
       'cudnn', opts.cudnn) ;
   
   
   % extracts dzdrow-col 
   dzdx0 = res_c(1).dzdrow;
   dzdx1 = res_c(1).dzdcol;
   if isNormalize
   % finds dzdrow-col with more than 1 pixel shift
   GTO = find(  (dzdx0 .^ 2 + dzdx1 .^ 2)  > 1/32);
   % normalize the dzdrow-col
   R0 = sqrt(dzdx0.^2 + dzdx1.^2);
   dzdx0(GTO) = dzdx0(GTO)./R0(GTO);
   dzdx1(GTO) = dzdx1(GTO)./R0(GTO);
   end
   % step
   dzdx0  = dzdx0 * atten_LR;
   dzdx1 = dzdx1 * atten_LR;
   % fits the new center into the history
   newCentHist = CentHist;
   newCentHist(1,1,1,:) = squeeze(CentHist(1,1,1,:)) - dzdx0;
   newCentHist(1,2,1,:) = squeeze(CentHist(1,2,1,:)) - dzdx1;
   if isFliped
       CentHist(1,2,1,:) = 1- newCentHist(1,2,1,:);
   else
       CentHist(1,:,1,:) = newCentHist(1,:,1,:);
   end
   net.layers{1}.centers = newCentHist;
   imdb.images.centerHist(:,:,:,batch) = gather(CentHist);
   %[im,orIm, labels,CentHist,newCent,polHist,imdb] = getBatch(imdb, batch,net,opts.usePolar,opts.useGmm) ;
   
        
end
    