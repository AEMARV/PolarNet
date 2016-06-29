function createHistVids(OutPath,netCount)
netName = 'net-epoch-';
netExt = '.mat';

for i = 1 : netCount
    Net = (load(fullfile(OutPath,[netName,int2str(i),netExt])));
    Net = Net.net;
    weights = justLayersHasWeight(Net);
    Frame=extractHists(weights);
    if i ==1 
        Movie = zeros(size(Frame,1),size(Frame,2),3,netCount);
    end
    Movie(:,:,:,i) = im2double(Frame);
    
end
frames2mov(Movie);
end
function Layers = justLayersHasWeight(net)
    Layers = cell(0,2);
    for i = 1 : numel(net.layers)
        l = net.layers{i};
        if numel(l.weights) ~= 0
            Layers(end+1,1) = {l.weights};
            Layers{end,2} = i;
        end
    end
end
function Hists = extractHists(weights)
    CurFig = figure('Position',[1,1,1000,500],'Visible','off');
    layerNum = size(weights,1);
    for i = 1 : size(weights,1)
        weis = weights{i,1};
        weis_mul = weis{1};
        weis_bias = weis{2};
         subplot(2,layerNum,sub2ind([layerNum,2],i,1));histogram(weis_mul(:),'Normalization','pdf');...
             title(['layer-',int2str(weights{i,2}),'-weights']);
         subplot(2,layerNum,sub2ind([layerNum,2],i,2));histogram(weis_bias(:),'Normalization','pdf');...
             title(['layer-',int2str(weights{i,2}),'-bias']);
         
        
    end
    F = getframe(CurFig);
    [Hists,Map] = frame2im(F);
   % Hists = ind2rgb(X,Map);
    close(CurFig);
end
function Mov = frames2mov(frames,OutPath)
Mov = immovie(frames);
implay(Mov);



end