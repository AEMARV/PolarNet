function [ resip1 ] = vl_nndropchannel( resi,resip1,dzdy,droprate )
CHANNELNUM = size(resi.x,3);
BATCH = size(resi.x,4);
if isempty(dzdy);doder = false ;else doder = true;end
if ~doder % forward pass
   droped = gpuArray.rand(1,1,CHANNELNUM,BATCH);
   Mask = droped > droprate;
   
   resip1.x = bsxfun(@times,Mask,resi.x);
   resip1.aux = Mask;
else % backward
    % resip1 is dzdx
    resip1 = bsxfun(@times,resip1.aux,dzdy);
end

end

