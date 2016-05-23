function  imdb = saveLoadDataParam(expDir,imdb,id,isSave)
%   function [ centers ] = saveLoadCenter( expDir , centers , id , isSave)
%   saves or loads the centers into .mat file in expDir directory
%   id is usually the epoch number or any id integer number specified
%   isSave specifies wether it will save or load the centers
DataParam = imdb.DataParam;
    
    if isSave
    save([expDir,sprintf('/DataParam%d.mat',id)],'DataParam');
    else
       load([expDir,sprintf('/DataParam%d.mat',id)],'DataParam'); 
       imdb.DataParam = DataParam;
    end
    
end

