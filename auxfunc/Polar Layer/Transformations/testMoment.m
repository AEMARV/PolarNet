close all
usePolar = false;
degree = 1;
im = imread('p.jpg');
% im = imread('circle.png');
im = im2double(im);
im = rgb2gray(im);
if usePolar
    % polar Parameters
    %-----------------------------------------
    opts = updateOptsPolar('type',0);
    opts.typePolar = 0;
    opts.upSampleRate=1;
    opts.DownSampleRate = 0;
    %------------------------------------------
    opts.filterSigma = single(10/3);
    img = gpuArray(im);
    center = gpuArray([size(img,1)/2,size(img,2)/2]);
    imresg = im2polar(img,center,opts);
    
    imres = gather(imresg);
else
    imres = im;
end
momentImage = im2moment(imres,degree);
%% DCT Transform
% 2D dct per row
dct2dMomentImage = dct2(momentImage);
dct2dImage = dct2(imres);
%% Compression
figure;
for maskIndex = [10:6:96,96]
    maskIndex
% maskIndex = 6;
% compression mask---------------------
mask= zeros(size(dct2dImage));
mask(1:maskIndex,1:end)=1;
%--------------------------------------
dct2dMomentImageCompressed = dct2dMomentImage.*mask;
dct2dImageCompressed = dct2dImage.*mask;
% reconstruct
imageCompressedReconstructed = idct2(dct2dImageCompressed);
momentImageCompressedReconstructed = idct2(dct2dMomentImageCompressed);
%% demoment
imageDeconsMoment = moment2im(momentImageCompressedReconstructed,degree);
%% Plot

subplot(2,2,1);
imshow(imres,[]);
title('original Image');
subplot(2,2,2);
imshow(momentImage/max(momentImage(:)))
title('Normalized Moment Image');
subplot(2,2,4);
imshow(imageDeconsMoment);
title('reconstructed Image with Moments');
subplot(2,2,3);
imshow(imageCompressedReconstructed)
title('reconstructed low passed original image');
pause(.5);
% figure;
% subplot(3,5,1);
% plot(1:size(momentImage,2), momentImage(end,:,1))
% subplot(3,5,2);
% plot(1:size(momentImage,2), momentImage(50,:,1))
end
figure;
subplot(2,1,1);
imshow(dct2dMomentImage);
title('Moment Dct');
subplot(2,1,2);
imshow(dct2dImage);
title('image dct');