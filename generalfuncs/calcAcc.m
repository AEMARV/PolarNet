function [ Pushacc ] = calcAcc( dzdw, rhoi,g,v )
%function [ Pushacc ] = calcAcc( dzdw, rhoi,g )
%   Detailed explanation goes here
theta = atan(dzdw);
pushingacc = -g .* sin(theta);
stopingacc = -sign(v) .* g .* cos(theta) .* rhoi;
Pushacc = pushingacc + stopingacc;
Pushacc = Pushacc .* cos(theta);

end

