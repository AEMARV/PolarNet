function  saveCentHist  = saveLoadCenter(expDir,saveCentHist,id,isSave)
%   function [ centers ] = saveLoadCenter( expDir , centers , id , isSave)
%   saves or loads the centers into .mat file in expDir directory
%   id is usually the epoch number or any id integer number specified
%   isSave specifies wether it will save or load the centers
    if isempty(saveCentHist)&& isSave
        error('centers are empty')
    end
    if isSave
    save([expDir,sprintf('/cents%d.mat',id)],'saveCentHist');
    else
       load([expDir,sprintf('/cents%d.mat',id)],'saveCentHist'); 
    end
end

