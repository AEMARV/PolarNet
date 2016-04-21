close all;
testCircle = 1;
testGradient = 1;
%circles = gpuArray(im2single(imread('circles.PNG')));
[x,y] = meshgrid(1:100,1:100);
circles = gpuArray(single(sqrt((x-50).^2 + (y-50).^2)));
batch = [1,3,5,10,100];
centers =  repmat([0.5,0.5],5,1);
imageBatch = images.data(:,:,:,batch);
imageBatch = gpuArray(single(imageBatch));
if testGradient == 1
    resi = struct('x',[],'dzdx',[]);
    resip1 = resi;
    resi.x = circles;
    opts.typePolar = 0;
opts.upSampleRate = 16;
opts.filterSigma = 32/3;
opts.extrapval = nan;
opts.kernel = gpuArray(single(fspecial('gaussian',ceil(opts.filterSigma *3),opts.filterSigma)));
   layer = createPolarLayer(opts);
   centers1 = [0.5,0.5];
   centers2 = [0.5,0.5];
   layer.centers = gpuArray([0.5,0.5]);
   
   [resip1original] = layer.forward(layer,resi,resip1)
else

if testCircle == 1
    imageBatch = circles;
    batch = 1;
    centers = [0.5,0.5];
end
opts.typePolar = 0;
opts.upSampleRate = 16;
opts.filterSigma = 32/3;
opts.extrapval = nan;
opts.kernel = gpuArray(single(fspecial('gaussian',ceil(opts.filterSigma *3),opts.filterSigma)));
Polarized = pol_transform(imageBatch,gpuArray(single(centers)),opts);

for i = 0 : numel(batch)-1
    
    subplot(2,numel(batch),i+1);imshow(Polarized(:,:,:,i+1),[]);title('Polar');
    subplot(2,numel(batch),numel(batch)+i+1); imshow(imageBatch(:,:,:,i+1),[]);title('original')
end
end