function imdb = createDataParam( imdb)
%% NewImdb = createDataParam(imdb) 
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
% NewImdb which has another field DataParam (struct array)
SIZE = size(imdb.images.data,4);
DataParam = struct(...
    'rmin', num2cell(zeros(1,SIZE)), ...
    'rmax', num2cell(ones(1,SIZE)), ...
    'theta0', num2cell(zeros(1,SIZE)), ...
    'row0', num2cell(zeros(1,SIZE)), ...
    'col0', num2cell(zeros(1,SIZE)));
    
imdb.DataParam = DataParam;
end

