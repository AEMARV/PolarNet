function imdb = createCentHist( imdb ,center_num)
%% NewImdb = createCentHist(imdb,center_num) 
% createCentHist attaches center history to the image database. 
% initially each entry has mean of 0.5,0.5 and sigma of 1.5,1.5
% ------------------------------------------------------------------------
% INPUTS: 
% imdb is a struct with fields 
% data: 
% a 4d matrix with X*Y*D*N dimensions where
% X : image height
% Y : image width
% D : number of channels in image
% N : dataset size
% label:
% 1*N array containing label numbers
% set: 
% 1*N array which shows whether the image is train,val or test
% ------------------------------------------------------------------------
% OUTPUTS:
% NewImdb which has another field centerHist of size 2*2*M*N
% M : the number of centers to be saved
% N : dataset size

init_cent_data = [0.5,0.5;1.5,1.5];
init_mult_cent = repmat(init_cent_data,[1,1,center_num]);
every_cent = repmat(init_mult_cent,[1,1,1,size(imdb.images.data,4)]);
imdb.images.centerHist = every_cent;
imdb.images.centerPol = single(nan *ones([size(imdb.images.data(:,:,:,1)),center_num,size(imdb.images.data,4)]));
end

