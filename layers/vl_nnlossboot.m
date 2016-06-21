function [ Y ] = vl_nnlossboot( X,c,dzdy )
%VL_NNLOSSBOOT Summary of this function goes here
%   Detailed explanation goes here
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

if numel(c) == sz(4)
  % one label per image
  c = reshape(c, [1 1 1 sz(4)]) ;
end
if size(c,1) == 1 & size(c,2) == 1
  c = repmat(c, [sz(1) sz(2)]) ;
end

% one label per spatial location
sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(sz_, [sz(1) sz(2) sz_(3) sz(4)])) ;
assert(sz_(3)==1 | sz_(3)==2) ;

% class c = 0 skips a spatial location
mass = cast(c(:,:,1,:) > 0, 'like', c) ;
if sz_(3) == 2
  % the second channel of c (if present) is used as weights
  mass = mass .* c(:,:,2,:) ;
  c(:,:,2,:) = [] ;
end

% convert to indexes
c = c - 1 ;
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * max(c(:), 0)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

% compute softmaxloss
LabelMul = gpuArray.ones(size(X));
LabelMul(c_) = -1;
Xlabeled = X.*LabelMul;
Err = vl_nnrelu(Xlabeled);

%n = sz(1)*sz(2) ;
if nargin <= 2
    Y = sum(Err(:));
%   t = Xmax + log(sum(ex,3)) - reshape(X(c_), [sz(1:2) 1 sz(4)]) ;
%   Y = sum(sum(sum(mass .* t,1),2),4) ;
else
  Y = sign(LabelMul .* vl_nnstochrelu( Xlabeled)) .*dzdy;
 % Y = disposeSamples(mask,Y);

end

