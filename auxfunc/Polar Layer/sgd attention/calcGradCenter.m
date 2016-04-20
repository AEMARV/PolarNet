function [dzdrow,dzdcol ] = calcGradCenter(dzdx,cartImage,CentHist,opts)
% function [dzdrow,dzdcol ] = calcGradCenter(dzdx,cartImage,CentHist)
% calculates how to change the location of the center of the attention.
% 
% ---------------------------------------------------------------------
% throughout the description 
% M is the row size of input images
% N is the col size of input images
% C is the number of channels of input images
% B is the batch size
% ---------------------------------------------------------------------
% Inputs: 
%
% dzdx : is the gradient of loss/ input image in polar coordinates. 
% dzdx is a M*N*C*B matrix
%   
% cartImage : input images in cartesian coordinates
% cartImage is a M*N*C*B
%
% CentHist : a tensor containing all the previous centers
% CentHist is a 2*2*1*B
%
% opts : a struct containing options for calculating polar transform
% --------------------------------------------------------------------
% Outputs:
% dzdrow shows the change to be done row location of the center
% dzdrow is of size B*1
%
% dzdcol shows the change to be done on col location of the center
% dzdcol is of size B*1
% ====================================================================

BATCH_SIZE = size(cartImage,4);
[gr,gc] = imGradient(cartImage,opts);
% gr gc are the same size as cartImage
if size(CentHist,4)>1
centers = squeeze(CentHist(1,:,1,:))';
else
centers = CentHist(1,:);
end
% centers are B*2 where first col is row and sec col is col coordinates
Polarized_grad = pol_transform(cat(4,gr,gc),cat(1,centers,centers),opts);
% Polarized_grad is of size M*N*C*(B*2) which has the row grad in the first
% B and col grad in the second B
Polar_grad_row = Polarized_grad(:,:,:,1:BATCH_SIZE);
% Polar_grad_row is of size M*N*C*B has polarized version of the gradient
% along the rows central at centHist
Polar_grad_col = Polarized_grad(:,:,:,BATCH_SIZE+1:end);
% Polar_grad_col is of size M*N*C*B has polarized version of the gradient
% along the rows central at centHist
el_times = dzdx .* Polar_grad_row;
% el_times is the element wise product of dzdx and Polar_grad_row
dzdrow = sum(sum(sum(el_times,1),2),3);
% dz drow should be of size B*1
dzdrow = squeeze(dzdrow);
el_times = dzdx .* Polar_grad_col;
% el_times is the element wise product of dzdx and Polar_grad_row
dzdcol = sum(sum(sum(el_times,1),2),3);
dzdcol = squeeze(dzdcol);
% dzdcol should be of size B*1

end
function [gr,gc] = imGradient(cartImage,opts)
% function [gr,gc] = imGradient(cartImage)
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
upSampleRate = opts.upSampleRate;
    useSobel = false;
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
        grgc = vl_nnconv(X,F,[],'Pad',1,'CuDNN');
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
        gr = vl_nnconv(X,Sobrow,[],'Pad',[0,1,0,0]);
        gr = reshape(gr,RowS,ColS,ChS,BatchS);
        gc = vl_nnconv(X,Sobcol,[],'Pad',[0,0,0,1]);
        gc = reshape(gc,RowS,ColS,ChS,BatchS);
    end
    
end

