function [movie,isTrain] =  createVideos(varargin)
% function movie =  createVideos(varargin)
%
%
%
% createVideos function gets the path to nets saved and the imdb and
% creates the videos of evolution of the network in polar domain.
% createVideos(varargin) and outputs a M*N*C*B*E tensor where M and N 
% represent the number of rows and columns of the images which is set by
% videoRes parameter (rectangular), C is the number of channels, B is the
% number of images, E is the epoch number it also outputs the set each
% image belongs to (Test/Train).
%
%
%=========================================================================
%INPUTS:
% createVideos does not have a fixed number of inputs. the function
% operates just on a set of parameters
%
% Parameters (default values):
% 
%
% dataBaseName = 'cifar'   :    name of the dataBase
%-------------------------------------------------------------------------
% imdbPath = nan           :    path to imdb created for the networks
% which is usually in the networks path
%-------------------------------------------------------------------------
% netPath = nan            :    path to the networks
%-------------------------------------------------------------------------
% imagePath = nan          :    path to the dataBase original files
%-------------------------------------------------------------------------
% movieOutPathBase = nan   :    path to where the movies are stored
%-------------------------------------------------------------------------
% videoRes = nan           :    integer defining the upsampling resolution
%-------------------------------------------------------------------------
% numberofVideos= nan      :  
%-------------------------------------------------------------------------
% BatchSize = 1024         :    this parameter does not affect the output
% except that at high numbers the gpu may not have enough memory
%-------------------------------------------------------------------------
% movieBaseName = [opts.dataBaseName,'_evolve'] : name of the movies
%-------------------------------------------------------------------------
% numEpoch = findLastCheckpoint(opts.netPath)   : number of epochs to create videos   
%-------------------------------------------------------------------------
%=========================================================================
%OUTPUTS
%
%movie: is a M*N*C*B*E tensor representing evolution of image uncertainty
%and eye move in different epochs
%-------------------------------------------------------------------------
%isTrain : shows wheter bth image in movie(:,:,:,b,:) is training image or
%test image,  true for training , false for test image
%-------------------------------------------------------------------------
opts.dataBaseName = 'cifar';
opts.imdbPath = nan;
opts.netPath = nan;
opts.imagePath = nan;
opts.movieOutPathBase = nan;
opts.movieBaseName = [opts.dataBaseName,'_evolve'];
opts.numEpoch = nan;
opts.videoRes = nan;
opts.numberofVideos= nan;
opts.BatchSize = 1024;
opts.videoFrameRate = 4;
opts = vl_argparse(opts,varargin);
if isnan(opts.numEpoch)
opts.numEpoch = findLastCheckpoint(opts.netPath);
end
% loading imdb
imdb = load(fullfile(opts.imdbPath,'imdb.mat'));
TrainIndices = find(imdb.images.set == 1 | imdb.images.set ==2);
TrainIndexLimit = [min(TrainIndices),max(TrainIndices)];
TestIndices = find(imdb.images.set == 3);
TestIndexLimit = [min(TestIndices),max(TestIndices)];
% selecting indices
TrainIndices = generateIndex(opts.numberofVideos,TrainIndexLimit);
TestIndices = generateIndex(opts.numberofVideos,TestIndexLimit);
isTrain = [TrainIndices>0 , TestIndices < 0];
indices = [TrainIndices,TestIndices];
procImage = imdb.images.data(:,:,:,indices);
labels = imdb.images.labels(indices);
fn = getFetchImageFunc('cifar');
pureImage = fn(indices,opts.imagePath);
[~,movie] = detect_radial(pureImage,procImage,indices,labels,opts);
TrainMovie = movie(:,:,:,isTrain,:);
fullPath = fullfile(opts.movieOutPathBase,'Train');
pasteMovie(TrainMovie,opts.videoFrameRate,fullPath,[opts.movieBaseName,'_TRAIN']);
TestMovie = movie(:,:,:,~isTrain,:);
fullPath = fullfile(opts.movieOutPathBase,'Test');
pasteMovie(TestMovie,opts.videoFrameRate,fullPath,[opts.movieBaseName,'_TEST']);
end

function fn = getFetchImageFunc(dataBaseName)
    switch dataBaseName
        case {'cifar'}
    fn = @(x,y)cifarRet(x,y);
        otherwise
            error('DataBase fetch function is not implemented');
    end
end
function Image = cifarRet(indices,dataBasePath)
% retrieves original images from dataBasePath
    SampleIneachSubdb = 10000;
    FileNumber = 6;
    Train_indicesRelative = cell(1,FileNumber);
    TestBatchNum = 6;
    DataSetName = 'cifar';
    ImageSize = [32,32,3];
    
    ImageCount = numel(indices);
    Image = zeros([ImageSize,ImageCount]);
    Index = 1;
    for i = 1:FileNumber
        % 
        BaseIndex = (i-1)*SampleIneachSubdb;
        ImaginaryIndex = find(indices <= i*SampleIneachSubdb &...
            indices > (i-1)*SampleIneachSubdb);
        IthSubDb = indices(ImaginaryIndex);
        IthSubDb_relative = IthSubDb - BaseIndex;
           
        Train_indicesRelative{i} = IthSubDb_relative;
        if numel(IthSubDb) <1
            continue;
        end
        
    if i ~= TestBatchNum
    imdb = load(fullfile(dataBasePath,['data_batch_',int2str(i),'.mat']));
    else
    imdb = load(fullfile(dataBasePath,['test_batch','.mat']));
    end
    data = imdb.data;
    selectData = data(IthSubDb_relative,:);
    EndIndex = Index + numel(IthSubDb) -1;
    Image(:,:,:,ImaginaryIndex) = flatImage(selectData,DataSetName);
    Index = EndIndex +1;
    end
end
function Image = flatImage(data,dataset)
        switch dataset
            case 'cifar'
        imagePure = data;
        imagePure = imagePure';
        imagePure = permute(reshape(imagePure,32,32,3,[]),[2,1,3,4]);
        imagePure = im2double(imagePure);
        Image = imagePure;
            otherwise
                error('dataSet no implemented');
        end
        
    
end
function Indices = generateIndex(count,Limit)
    RANDOMS = rand(1,count);
    scope = Limit(2) - Limit(1);
    if scope< 0,error('start index is bigger than end Index');end
    RANDOMS = floor(RANDOMS * scope); % between 0 to scope
    Indices = RANDOMS +Limit(1); %between Limit1 to scope + Limit1 (Limit 2)
end
function pasteMovie(movie,videoFrameRate,path,baseName)
%function pasteMovie(movie,videoFrameRate,path,baseName)
    mkdir(path);
    for ImageNum = 1 : size(movie,4)
        
        v = VideoWriter(fullfile(path,[baseName,'_',int2str(ImageNum),'.avi']),'Uncompressed AVI');
        v.FrameRate = videoFrameRate;
        open(v);
        for epochNum = 1: size(movie,5)
            writeVideo(v,movie(:,:,:,ImageNum,epochNum));
        end
        close(v);
    end
end
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end