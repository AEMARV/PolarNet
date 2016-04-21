imdbPath = './data/cifar-lenet/'
netPath = imdbPath;
imagePath = './data/cifar/cifar-10-batches-mat/'
movieOutPathBase = './exp/MOVIEOUT/'
movieBaseName = '';
numEpoch = 27;
videoRes = 32;
numberofVideos= 20;

load([imdbPath,'imdb.mat']);
for i = 1 : numberofVideos
if mod(i-1,2) == 0
    movieName = [movieBaseName,'train'];
    train = 1;
else
    movieName = [movieBaseName,'test'];
    train = 0;
end
detect_radial(train,images,imagePath,movieOutPathBase,netPath,[movieName,int2str(i)],numEpoch,videoRes);
end

