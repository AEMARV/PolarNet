function [x1,v1] = calcnextpos( x0,v0,a0,dt)
%
%function [ output_args ] = calcnextpos( x0,v0,a0,dt)
dt = gpuArray(dt);
stopt  = -v0./a0;
stopt(a0 == 0 ) = -1;
    stopt(stopt > dt | stopt <= 0) = dt;

x1 = (1/2 .* a0 .* (stopt.^2)) + (v0 .* stopt) + x0;
v1 = a0.*stopt + v0;

end

