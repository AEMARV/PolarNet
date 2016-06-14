function runAll(cont)
if nargin ==0 
    cont = false;
end
cnn_cifar('train',struct('gpus',1),'expDir','./results','continue',cont);
end