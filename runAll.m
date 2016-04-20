function out = runALL()
cnn_cifar('train',struct('gpus',1),'expDir','./results');
end