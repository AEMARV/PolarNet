function [mov,concatMov] = detect_radial(imagePure,procImage,indices,labels,opts)
% function [a,c,mov] = detect_radial(imdbPath,imagePath,MovieOutPath,netPath,MovieName,numEpochs,videoRes)
CHANNEL_NUM = size(procImage,3);
samplingSize = opts.videoRes;
BatchSize = opts.BatchSize;
numEpochs = opts.numEpoch;
netPath = opts.netPath;
ImageCount = numel(indices);
%%% TODO :::: imresize does not work with multiple images
assert(samplingSize == size(imagePure,1),'resizing is not implemented\n...please set the video resolution to dataset original size')
imagePure = imresize(imagePure,[samplingSize,samplingSize]);
procImage = gpuArray(procImage);
r0 = 0.2;
c0 = 0.1;
a = zeros(samplingSize,samplingSize);
c = zeros(samplingSize,samplingSize);
z= zeros(samplingSize,3*samplingSize,CHANNEL_NUM);
% create movie
%v = VideoWriter([opts.movieOutPathBase,MovieBaseName,'.avi'],'Uncompressed AVI');
%v.FrameRate = videoFrameRate;
%open(v);
r0 = 1: samplingSize;
c0 = 1 : samplingSize;
r0 = r0/samplingSize;
c0 = c0 / samplingSize;
[c0,r0] = meshgrid(c0,r0);
centersC = gpuArray(cat(2,r0(:),c0(:)));
mov = zeros([size(procImage),numEpochs]);
concatMov = zeros(samplingSize,3*samplingSize,CHANNEL_NUM,ImageCount,numEpochs);
%procImage = repmat(procImage,1,1,1,BatchSize);
for ep = 1 :numEpochs
    load(fullfile(netPath,['net-epoch-',int2str(ep),'.mat']));
    load(fullfile(netPath,['cents',int2str(ep),'.mat']));
    net_gpu = vl_simplenn_move(net,'gpu');
    net_gpu.layers{end}.class = ones(1,BatchSize);
    for j = 1: ImageCount
        imageIndex = indices(j);
        chosenCenter = centerHist(1,:,1,imageIndex);
        chosenCenter = chosenCenter * samplingSize;
        cur_image = procImage(:,:,:,j);
        cur_image = repmat(cur_image,1,1,1,BatchSize);
        for k = 1:BatchSize: numel(r0)-1
            %im1P = pol_transform(im1,centersC(k:k+BatchSize-1,:));
            net_gpu.layers{1}.centers = centersC(k:k+BatchSize-1,:);
            net_gpu.layers{1}.randomRotate =false;
            res = vl_simplenn(net_gpu,cur_image);
            [m,MAXIND] = max(res(end-1).x,[],3);
            ctemp = 1-AM_entropy(sigmoid(gather(res(end-1).x),'sigmoid'),1);
            [rtemp,coltemp] = ind2sub(size(c),k:k+BatchSize-1);
            c(k:k+BatchSize-1) = ctemp(:);
            a(k:k+BatchSize-1) = gather(MAXIND(:));
        end
        assert(min(c(:))>=0 & max(c(:))<=1);
        c = c./(max(c(:)));
        %labels_image = z;
        
        z(1:samplingSize,2*samplingSize +1 :3* samplingSize,:)= imagePure(:,:,:,j);
        z(1:samplingSize,1:samplingSize,2) = c .* (a == labels(j));
        z(1:samplingSize,1:samplingSize,1) = c .* (a ~= labels(j));
        if floor(chosenCenter(1)) >= 1 && floor(chosenCenter(1)) <= 32 && floor(chosenCenter(2)) >= 1 && floor(chosenCenter(2)) <= 32
            if a(floor(chosenCenter(1)),floor(chosenCenter(2))) == labels(j);
                z(floor(chosenCenter(1)),floor(chosenCenter(2)),3) = 1;
                z(floor(chosenCenter(1)),floor(samplingSize+chosenCenter(2)),3) = 1;
            else
                z(floor(chosenCenter(1)),floor(chosenCenter(2)),3) = 1;
                z(floor(chosenCenter(1)),floor(chosenCenter(2)),2) = 1;
                z(floor(chosenCenter(1)),floor(2*samplingSize+chosenCenter(2)),3) = 1;
                z(floor(chosenCenter(1)),floor(2*samplingSize+chosenCenter(2)),2) = 1;
            end
        end
        z(1:samplingSize,samplingSize+1:2*samplingSize,1) = c;
        z(1:samplingSize,samplingSize+1:2*samplingSize,2:3) = 1;
        z(1:samplingSize,samplingSize+1:2*samplingSize,:) = hsv2rgb(z(1:samplingSize,samplingSize+1:2*samplingSize,:));
        assert(min(z(:))>=0 & max(z(:))<=1);
        z = min(z,1);
        z = max(z,0);
        mov(:,:,:,j,ep) = z(1:samplingSize,1:samplingSize,:);
        concatMov(:,:,:,j,ep) = z(:,:,:);
     %   writeVideo(v,z);
        z = zeros(size(z));
    end
    %figure; imshow(imresize(z,[size(im1cut,1),size(im1cut,2)]),[]);title('uncertainty');
    %labels_image = imresize(z,[size(im1cut,1),size(im1cut,2)]);
    %movieImages(:,:,:,i) = labels_image;
end
%close(v);
%mov = immovie(movieImages);
%implay(mov);

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