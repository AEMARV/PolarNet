[x,y] = meshgrid(1:32,1:32);
newNet = false;
circles = gpuArray(single(sqrt((x-16).^2 + (y-16).^2)));
circles = repmat(circles,1,1,3,8);
load('/home/student/Documents/MATLAB/matconvnet-master/data/cifar-lenet/net-epoch-10.mat');

opts.upSampleRate = double(8);
opts.filterSigma = single(2*opts.upSampleRate/3);
opts.kernel = single(fspecial('gaussian',ceil(double(opts.filterSigma *3)),double(opts.filterSigma)));

net.layers{1}.upSampleRate = opts.upSampleRate;
net.layers{1}.kernel = opts.kernel;
net.layers{1}.typePolar = 1;


[net.layers{end},~,~] = createUncertainLayer();
centers = [0.5,0.5;1,1];
centers = repmat(centers,1,1,1,8);
epsilone = 0.000001;
for i = 1 : 8
    if mod(i-1,2) ==0 
    centers(1,1,1,i) = centers(1,1,1,i) + floor((i-1)/2)*epsilone;
    else
        centers(1,2,1,i) = centers(1,2,1,i) + floor((i-1)/2)*epsilone;
    end
end
net.layers{1}.centers = centers;
net = vl_simplenn_move(net,'gpu');
res = vl_simplenn(net,circles,gpuArray(single(1)));
frow = res(end).x(1,1,1,1);
fcol = res(end).x(1,1,1,2);
frowpdelta = res(end).x(1,1,1,3);
fcolpdelta = res(end).x(1,1,1,4);
dzdx0emp = [(frowpdelta-frow)/epsilone,(fcolpdelta-fcol)/epsilone];
R = sqrt(sum(dzdx0emp(:).^2));
dzdx0emp = dzdx0emp./R
dzdrowanal = res(1).dzdrow;
dzdcolanal = res(1).dzdcol;
R = sqrt(dzdrowanal(1).^2 + dzdcolanal(1).^2);
dzdrowanal(1) = dzdrowanal(1)/R;
dzdcolanal(1) = dzdcolanal(1)/R;
dzdx0anal = [dzdrowanal(1),dzdcolanal(1)]


%net.layers.centers(1) = centers;