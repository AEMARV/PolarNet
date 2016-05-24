function err = checkDerivativeNumerically(f, x, dzdy)
%CHECKDERIVATIVENUMERICALLY  Check a layer's deriviative numerically
%   ERR = CHECKDERIVATIVENUMERICALLY(F, X, DX) takes the scalar function F,
%   its tensor input X and its derivative DX at X and compares DX to
%   a numerical approximation of the derivative returing their difference
%   ERR.



err = dx_numerical - dx ;

range = max(abs(dx(:))) * [-1 1] ;
T = size(x,4) ;
for t = 1:size(x,4)
  subplot(T,3,1+(t-1)*3) ; bar3(dx(:,:,1,t)) ; zlim(range) ;
  title(sprintf('dx(:,:,1,%d) (given)',t)) ;
  subplot(T,3,2+(t-1)*3) ; bar3(dx_numerical(:,:,1,t)) ; zlim(range) ;
  title(sprintf('dx(:,:,1,%d) (numerical)',t)) ;
  subplot(T,3,3+(t-1)*3) ; bar3(abs(err(:,:,1,t))) ; zlim(range) ;
  title('absolute difference') ;
end
