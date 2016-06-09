function runAll(cont)
cnn_cifar('train',struct('gpus',1),'expDir','./results','continue',cont);
end