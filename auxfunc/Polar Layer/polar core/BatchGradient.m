function [gr,gc] = BatchGradient(cartImage,padRowType,padColType)
% function [gr,gc] = BatchGradient(cartImage,padRowType,padColType)
% calculates the Sobel gradients of a multichannel batch of images
% --------------------------------------------------------------------
% INPUTS:
%
% cartImage :  is batch of images
% cartImage is a M*N*C*B
%
% OUTPUTS:
% 
% gr : gradients along the rows
    if nargin ==1 
    padColType = 0;
    padRowType = 0;
    end
    useSobel = true;
    if useSobel
        Sobrow = gpuArray(single([-1,-2,-1;0,0,0;1,2,1]));
        Sobcol = gpuArray(single([-1,0,1;-2,0,2;-1,0,1]));
        pad = 1;
        RowS = size(cartImage,1);
        ColS = size(cartImage,2);
        BatchS = size(cartImage,4);
        ChS = size(cartImage,3);
        F = cat(4,Sobrow,Sobcol);
        X = reshape(cartImage,RowS,ColS,1,ChS*BatchS);
        X = padarray(X,[0,1],padColType,'both');
        X = padarray(X,[1,0],padRowType,'both');
        %grgc = vl_nnconv(X,F,[],'Pad',1,'CuDNN');
        grgc = vl_nnconv(X,F,[],'CuDNN');
        % grgc is M*N*2*(B*CH) where dim 3 page 1 contains row grad MUST CHECK ???
        gr = reshape(grgc(:,:,1,:),RowS,ColS,ChS,BatchS);
        gc = reshape(grgc(:,:,2,:),RowS,ColS,ChS,BatchS);
        % gr gc must be M*N*C*B   MUSTCHECK
    else
        Sobrow = gpuArray(single([-1;1]));
        Sobcol = gpuArray(single([-1,1]));
        RowS = size(cartImage,1);
        ColS = size(cartImage,2);
        BatchS = size(cartImage,4);
        ChS = size(cartImage,3);
        %F = cat(4,Sobrow,Sobcol);
        X = reshape(cartImage,RowS,ColS,1,ChS*BatchS);
        gr = vl_nnconv(X,Sobrow,[],'Pad',[0,1,0,0],'CuDNN');
        gr = reshape(gr,RowS,ColS,ChS,BatchS);
        gc = vl_nnconv(X,Sobcol,[],'Pad',[0,0,0,1],'CuDNN');
        gc = reshape(gc,RowS,ColS,ChS,BatchS);
    end
    
end

