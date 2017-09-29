function z = surfaceGenerate( x,y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


if y<exp(0.15*x)-0.85 && x>exp(-0.15*y)-0.75
    z=1500;
elseif y<exp(0.15*x)-0.85 && x<exp(-0.15*y)-0.75
        if y>x
            z=10;
        else
            z=1500;
        end
elseif y>exp(0.15*x)-0.85 && x<exp(-0.15*y)-0.75
    z=10;
else
    z=400+1600*exp(-7*y)+x;
end

