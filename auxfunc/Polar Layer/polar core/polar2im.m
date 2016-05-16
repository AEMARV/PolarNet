function Zinterp = polar2im(M,opts)
%% im2polar converts an image from cartesian to polar coordinates
%
% function Zinterp = im2polar(M)
% M is the image which should be double
%--------------------------------------------------------------------------
% Center determines the center of polar coordinates, on input -1 it is the 
% center of image 
%--------------------------------------------------------------------------
 % generate fake image
 
 if (~isa(M,'gpuArray'))
 error('not gpuArray');
 
 end
 %M = im2single(M);
 plotOn = 1;
 
 
    SizeRow = size(M,1);
    SizeCol = size(M,2);
 % Determine the minimum and the maximum x and y values:
 %rmin = 0; tmin = -pi;
 %rmax = min(size(M,1)/2,size(M,2)/2); tmax = pi;
  rmin = -SizeRow/2; cmin = -SizeCol/2;
  rmax = SizeRow/2; cmax = SizeCol/2;
  rres = SizeRow;
  cres = SizeCol;
 % Define the resolution of the grid:
 %F = scatteredInterpolant(rho,theta,z,'linear','none');
 
 %Evaluate the interpolant at the locations (rhoi, thetai).
 %The corresponding value at these locations is Zinterp:
    
 [row,col] = meshgrid(gpuArray.linspace(rmin,rmax,rres),gpuArray.linspace(cmin,cmax,cres));

 %Zinterp = F(rhoi,thetai);
 %Zinterp(isnan(Zinterp)) = 0;
 radius =((SizeRow-1)* sqrt(row.^2 + col.^2)/(SizeRow/2)) +1;
 theta = atan(col./row);
 theta = theta + ((pi).*((col>0) .* (row <0))) -((pi) .* ((col<0) .*(row<0)));
 theta = (((theta +pi)/(2*pi))*(SizeCol-1)) +1 ;
 Zinterp =M;
 Zinterp(:,:,1) =  interp2(M(:,:,1), theta,radius,'linear',0)';
 if size(M,3)==3
 Zinterp(:,:,2) =  interp2(M(:,:,2),theta,radius,'linear',0)';
 Zinterp(:,:,3) =  interp2(M(:,:,3),theta,radius,'linear',0)';
 end
 
 if plotOn == 1
    subplot(1,2,1); imshow(M,[]) ; axis square
    subplot(1,2,2); imshow(Zinterp,[]) ; axis square
 end
end