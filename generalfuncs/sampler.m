function out = sampler(prob)

    out = prob>gpuArray.rand(size(prob));


end