function [ imr ] = imageRotate( im,theta,method )
%% function [ imr ] = imageRotate( im,theta,method )
% rotates single input image 
% -Theta is from 0 to 360
% -method:
%   'crop': crops the image in order to match the output image to input
%   image and fill the resulting values with zero
%   'mirror': crop the image in order to match the output image size and
%   fill with the mirror of image.
    switch method
        case 'crop'
            imr = imrotate(im,theta,'crop');
        case 'mirror'
            imr = imageRotateMirror(im,theta);
    end
            
            

end
function [imr]= imageRotateMirror(im,theta)
    temp = zeros(3*size(im,1),3*size(im,2),3); % creating 3 by 3 cells of image in order to create mirror
    fh = flip(im,2);
    fv = flip(im,1);
    fb= flip(fv,2); % flipping the image horizontally and vertically for images on the corner of 3x3 cell
    for row = 0:2
        for col = 0:2
            rIndex = 1+row*size(im,1):(row+1)*size(im,1); % indexes to change in the bigger image
            cIndex = 1+col*size(im,2):(col+1)*size(im,2);
            if(row==1)
                temp(rIndex,cIndex,:)=fh;
            else
                if(col==1)
                    temp(rIndex,cIndex,:)=fv;
                else
                    temp(rIndex,cIndex,:)=fb;
                end

            end
            if(row*col==1)
                temp(rIndex,cIndex,:) = im; 
            end

        end
    end
    temp=imrotate(temp,theta,'crop','bicubic');
    imr = temp(1+size(im,1):2*size(im,1),1+size(im,2):2*size(im,2),:);
end
