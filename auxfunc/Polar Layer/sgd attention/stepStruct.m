function stout =  stepStruct(st1,st2,LR)
stout = st1;
Learn = @(point,gradient,LR)point-(gradient.*LR);
Fnames = fieldnames(st1);
for i = 1:numel(Fnames)
    
    stout.(Fnames{i}) = Learn(st1.(Fnames{i}),st2.(Fnames{i}),LR);
    if strcmp(Fnames{i},'rmin')
        stout.(Fnames{i}) = max(stout.(Fnames{i}),0);
    end
    
end
Inds = find(stout.rmax - stout.rmin <0);
stout.rmax(Inds) = stout.rmin(Inds);
end