function [ ] = writeOptsxml(opts,fileName)
%WRITEOPTSXML Summary of this function goes here
%   Detailed explanation goes here
    wrap.OPTIONS = opts;
    struct2xml(wrap,fileName);

end

