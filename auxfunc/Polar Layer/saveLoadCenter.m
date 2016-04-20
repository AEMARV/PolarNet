function  imdb = saveLoadCenter(expDir,imdb,id,isSave)
%   function [ centers ] = saveLoadCenter( expDir , centers , id , isSave)
%   saves or loads the centers into .mat file in expDir directory
%   id is usually the epoch number or any id integer number specified
%   isSave specifies wether it will save or load the centers
centerHist = imdb.images.centerHist;
    if isempty(imdb.images.centerHist)&& isSave
        error('centers are empty')
    end
    if isSave
    save([expDir,sprintf('/cents%d.mat',id)],'centerHist');
    else
       load([expDir,sprintf('/cents%d.mat',id)],'centerHist'); 
       imdb.images.centerHist = centerHist;
    end
    
end

