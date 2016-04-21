function [a,c,mov] = detect_radial(training,imdb,imagePath,MovieOutPath,netPath,MovieName,numEpochs,videoRes)
% function [a,c,mov] = detect_radial(imdbPath,imagePath,MovieOutPath,netPath,MovieName,numEpochs,videoRes)
useCifar = 1;
images = imdb;
debug_polar_function = 0;
samplingSize = videoRes;
BatchSize = 1024;
debug = 0;
if ~useCifar
im1or_orSize = imread('catdog.jpg');
im1or_orSize = single(im1or_orSize);

fig =figure; imshow(im1or_orSize/255,[]);
rect = getrect(fig);
selRow = rect(2);
selCol = rect(1);
endSelcol = selCol + rect(3);
endSelrow = selRow + rect(4);

im1or= im1or_orSize(selRow:endSelrow,selCol:endSelcol,:);
im1cut = im1or;

fig2 = figure;
%% debug polar function
if debug_polar_function
for i = 1 : 10
[col,row] = getpts(fig);
col
row
testPol = pol_transform( gpuArray(im1or_orSize),[row/size(im1or_orSize,1),col/size(im1or_orSize,2)]);
testCart = polar2im(testPol,0);
figure(fig2);imshow(testCart/255,[]);

end
end
imshow(im1or/255,[]);
net.layers{13}.class = 1;
im1 = im2single(im1or);

im1NP  = imresize(im1,[samplingSize,samplingSize]);
% prepare the input
im1 = imresize(im1,[samplingSize,samplingSize]);
im1 =prepData(im1, dataMean, en,V,d2,n);
im1or = im1;

%end
end
if useCifar

  %  pickimage = (images.set==1);
    if training 
    imageIndex = floor(rand * 50000);
    load([imagePath,'/data_batch_',int2str(ceil(imageIndex/10000)),'.mat']);
    imageBatchNum = ceil(imageIndex/10000);
    imageSubIndex = imageIndex - (imageBatchNum-1)*10000;
    imagePure = data(imageSubIndex,:);
    imagePure = imagePure';
    imagePure = permute(reshape(imagePure,32,32,3),[2,1,3]);
    imagePure = im2double(imagePure);
    imagePure = imresize(imagePure,[samplingSize,samplingSize]);
    else
        imageIndex = floor(50000 + (rand * 10000));
        load([imagePath,'test_batch','.mat']);
        imageSubIndex = imageIndex - 50000;
        imagePure = data(imageSubIndex,:);
        imagePure = imagePure';
        imagePure = permute(reshape(imagePure,32,32,3),[2,1,3]);
        imagePure = im2double(imagePure);
        imagePure = imresize(imagePure,[samplingSize,samplingSize]);
        
    end
    im1 =images.data(:,:,:,imageIndex);
    sprintf('image number %d', imageIndex);
    im1cut = im1;
else
end
% end of preparation
im1 = gpuArray(im1);
r0 = 0.2;
c0 = 0.1;
a = zeros(samplingSize,samplingSize);
c = zeros(samplingSize,samplingSize);
z= zeros(samplingSize,2*samplingSize,3);
% create movie 
v = VideoWriter([MovieOutPath,MovieName,'.avi'],'Uncompressed AVI');
v.FrameRate = 4;
open(v);
r0 = 1: samplingSize;
c0 = 1 : samplingSize;
r0 = r0/samplingSize;
c0 = c0 / samplingSize;
[c0,r0] = meshgrid(c0,r0);
centersC = gpuArray(cat(2,r0(:),c0(:)));
im1 = repmat(im1,1,1,1,BatchSize);
for i = 1 :numEpochs
load([netPath,'net-epoch-',int2str(i),'.mat']);
load([netPath,'cents',int2str(i),'.mat']);
chosenCenter = saveCentHist(1,:,1,imageIndex);
chosenCenter = chosenCenter * samplingSize;
net_gpu = vl_simplenn_move(net,'gpu');
net_gpu.layers{end}.class = ones(1,BatchSize);
'load'
i

for k = 1:BatchSize: numel(r0)-1 
%im1P = pol_transform(im1,centersC(k:k+BatchSize-1,:));
net_gpu.layers{1}.centers = centersC(k:k+BatchSize-1,:);
res = vl_simplenn(net_gpu,im1);
    [m,i] = max(res(end-1).x,[],3);
    ctemp = 1-AM_entropy(sigmoid(gather(res(end-1).x),'sigmoid'),1);
    [rtemp,coltemp] = ind2sub(size(c),k:k+BatchSize-1);
    c(k:k+BatchSize-1) = ctemp(:);
    a(k:k+BatchSize-1) = gather(i(:));
end
c = c./(max(c(:)));   
%labels_image = z;
movieImages = zeros(size(im1cut,1),size(im1cut,2),3,numEpochs);
z(1:samplingSize,samplingSize +1 :2* samplingSize,:)= imagePure;
z(1:samplingSize,1:samplingSize,2) = c .* (a == images.labels(imageIndex));
z(1:samplingSize,1:samplingSize,1) = c .* (a ~= images.labels(imageIndex));
if floor(chosenCenter(1)) >= 1 && floor(chosenCenter(1)) <= 32 && floor(chosenCenter(2)) >= 1 && floor(chosenCenter(2)) <= 32  
if a(floor(chosenCenter(1)),floor(chosenCenter(2))) == images.labels(imageIndex);
z(floor(chosenCenter(1)),floor(chosenCenter(2)),3) = 1;
z(floor(chosenCenter(1)),floor(samplingSize+chosenCenter(2)),3) = 1;
else
    z(floor(chosenCenter(1)),floor(chosenCenter(2)),3) = 1;
    z(floor(chosenCenter(1)),floor(chosenCenter(2)),2) = 1;
    z(floor(chosenCenter(1)),floor(samplingSize+chosenCenter(2)),3) = 1;
    z(floor(chosenCenter(1)),floor(samplingSize+chosenCenter(2)),2) = 1;
end
end
z = min(z,1);
z = max(z,0);
writeVideo(v,z);
z = z.* 0;
%figure; imshow(imresize(z,[size(im1cut,1),size(im1cut,2)]),[]);title('uncertainty');
%labels_image = imresize(z,[size(im1cut,1),size(im1cut,2)]);
%movieImages(:,:,:,i) = labels_image;
end
close(v);
%mov = immovie(movieImages);
%implay(mov);
mov = 0;
fprintf('\n image number %d', imageIndex);
end
function im1 = prepData(im1, dataMean, en,V,d2,n)
im1 = im1 - dataMean;


%if opts.contrastNormalization
  im1 = im1 - mean(im1(:)) ;
  %n = std(z,0,1) ;
  ncurrentimage = std(im1(:));
  im1 = bsxfun(@times, im1, mean(n) ./ max(ncurrentimage, 40)) ;
  im1 = reshape(im1, 32, 32, 3) ;
%end

%if opts.whitenData
  % the scale is selected to approximately preserve the norm of W
  im1 = reshape(im1,[],1);
  im1 = V*diag(en./max(sqrt(d2), 10))*V'*im1 ;
  im1 = reshape(im1, 32, 32, 3) ;
end