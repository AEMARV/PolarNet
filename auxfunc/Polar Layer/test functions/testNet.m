newimdb = createCentHist(imdb,3);
val_batch = newimdb.images.set ==3;
net = vl_simplenn_move(net, 'gpu') ;
labels = imdb.images.labels(1,val_batch) ;
net.layers{end}.class = labels;
for i = 1:10
    centers=genCenter(newimdb.images.centerHist(:,:,:,val_batch));
    images = newimdb.images.data(:,:,:,val_batch);
images = pol_transform(gpuArray(images),centers);
res = vl_simplenn(net,images);
newHist = addNewCenters(res(13).x,newimdb.images.centerHist(:,:,:,val_batch),centers,0,1);
newimdb.images.centerHist = updateHistImdb(newimdb.images.centerHist,newHist,find(val_batch));
end