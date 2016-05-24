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
    'rmin', zeros(1,SIZE,'single')', ...
    'rmax', ones(1,SIZE,'single')', ...
    'theta0', zeros(1,SIZE,'single')', ...
    'row0', zeros(1,SIZE,'single')', ...
    'col0', zeros(1,SIZE,'single')');
    
imdb.DataParam = DataParam;
end

