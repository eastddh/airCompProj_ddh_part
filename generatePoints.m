function [diff_Lat,Lats,diff_Lon,Lons] = generatePoints(center_lat, center_lon, R, numPoints)
%GENERATEPOINTS Summary of this function goes here
%   Detailed explanation goes here
% R is the maximum radius from the center, meters

%Lats = zeros(numPoints,1);
%Lons = zeros(numPoints,1);
diss = rand(numPoints,1)*R;
thetas = rand(numPoints,1)*2*pi;
earth_r = 6371000;
diff_Lat = diss .* sin(thetas)/ (2*pi*earth_r/360);
Lats = diff_Lat+center_lat;

diff_Lon = diss .* cos(thetas)./(2*pi*earth_r*cos(Lats/180*pi)/360);
Lons = diff_Lon+center_lon;
end

