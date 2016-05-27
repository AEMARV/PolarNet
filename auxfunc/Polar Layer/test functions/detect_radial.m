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
r0 = linspace(-1,1,samplingSize);
c0 = linspace(-1,1,samplingSize);
[c0,r0] = meshgrid(c0,r0);
DataParam_per_loc.row0 = r0(:);
DataParam_per_loc.col0 = c0(:);
mov = zeros([size(procImage),numEpochs]);
concatMov = zeros(samplingSize,3*samplingSize,CHANNEL_NUM,ImageCount,numEpochs);
%procImage = repmat(procImage,1,1,1,BatchSize);
for ep = 1 :numEpochs
    netLoaded = load(fullfile(netPath,['net-epoch-',int2str(ep),'.mat']));
    net = netLoaded.net;
    DataParamLoaded = load(fullfile(netPath,['DataParam',int2str(ep),'.mat']));
    DataParam = DataParamLoaded.DataParam;
    net_gpu = vl_simplenn_move(net,'gpu');
    net_gpu.layers{end}.class = ones(1,BatchSize);
    imdbFake.DataParam = DataParam;
    for j = 1: ImageCount
        imageIndex = indices(j);
        chosenCenter = [DataParam.row0(imageIndex),DataParam.col0(imageIndex)];
        ThisDataParam = getDataParamImdb(imdbFake,imageIndex,false);
        ThisDataParam_rep = extendDataParam(ThisDataParam,BatchSize);
        cur_image = procImage(:,:,:,j);
        cur_image = repmat(cur_image,1,1,1,BatchSize);
        for  k = 1:BatchSize: numel(r0)-1
            %im1P = pol_transform(im1,centersC(k:k+BatchSize-1,:));
%             net_gpu.layers{1}.centers = centersC(k:k+BatchSize-1,:);
%             net_gpu.layers{1}.randomRotate =false;
%             net_gpu.layers{1}.rowColShift =repmat(chosenRot,[1,BatchSize]);
            ThisDataParam_rep.row0 = r0(k:k+BatchSize-1)';
            ThisDataParam_rep.col0 = c0(k:k+BatchSize-1)';
            res = vl_simplenn(net_gpu,cur_image,ThisDataParam_rep);
            [m,MAXIND] = max(res(end-1).x,[],3);
            ctemp = 1-AM_entropy(sigmoid(gather(res(end-1).x),'sigmoid'),1);
            [rtemp,coltemp] = ind2sub(size(c),k:k+BatchSize-1);
            c(k:k+BatchSize-1) = ctemp(:);
            a(k:k+BatchSize-1) = gather(MAXIND(:));
        end
        assert(min(c(:))>=0 & max(c(:))<=1);
        c = c./(max(c(:)));
        %labels_image = z;
        image_shaped = drawIndicators(imagePure(:,:,:,j),ThisDataParam);
        z(1:samplingSize,2*samplingSize +1 :3* samplingSize,:)= image_shaped;
        z(1:samplingSize,1:samplingSize,2) = c .* (a == labels(j));
        z(1:samplingSize,1:samplingSize,1) = c .* (a ~= labels(j));
        
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

function DataParam= extendDataParam(DataParam,SIZE)
    Fnames = fieldnames(DataParam);
    for i = 1 : numel(Fnames)
        FieldVal = DataParam.(Fnames{i});
        ExtendVal = repmat(FieldVal,SIZE,1);
        assert(numel(ExtendVal) == SIZE);
        DataParam.(Fnames{i}) = ExtendVal;
    end
end
function out =  drawIndicators(ImagePure,DataParam)
    COLOR = double([1,0,0]);
    ImagePure = gather(ImagePure);
    rmin = gather(DataParam.rmin)*16;
    rmax = gather(DataParam.rmax)*16;
    theta0 = gather(DataParam.theta0);
    y0 = gather(DataParam.row0)*16 + 16;
    x0 = gather(DataParam.col0)*16 + 16;
    
    InsertObj = vision.ShapeInserter;
    InsertObj.BorderColor = 'Custom';
    InsertObj.CustomBorderColor = COLOR;
    InsertObj.Shape = 'Circles';
    centers = zeros(2,3);
    centers(:,3) = [rmin;rmax];
    centers(:,1) = [x0;x0];
    centers(:,2) = [y0;y0];
    out = step(InsertObj,ImagePure,centers);
    InsertObj.release();
    InsertObj.Shape = 'Lines';
    line = [x0,y0,rmax*sin(theta0)+x0,rmax*cos(theta0)+y0];
    out = step(InsertObj,out,line);
    
    
end