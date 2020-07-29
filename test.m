close all; clear all;
tx = txsite('Latitude',42.3001, ...
   'Longitude',-71.558317, ...
   'TransmitterFrequency',2.5e9);
%show(tx);

numPoints = 500;
R = 10000;% meter
%generate RXs
[diff_Lat,Lats,diff_Lon,Lons] = generatePoints(tx.Latitude, tx.Longitude, R, numPoints);
rxSen = -90; %dbm
rxs = rxsite(...
    'Latitude',Lats, ...
    'Longitude',Lons, ...
    'Antenna',design(dipole,tx.TransmitterFrequency), ...
    'ReceiverSensitivity',rxSen);
pm = propagationModel('longley-rice','TimeVariabilityTolerance',0.7);
PLs = zeros(numPoints,1); %record pathloss
SSs = zeros(numPoints,1); % record received signal strength
eles = zeros(numPoints,1); % record elevation in meters
for i=1:numPoints
    PLs(i) = pathloss(pm,rxs(i),tx);
    SSs(i) = sigstrength(rxs(i),tx);
    eles(i) = elevation(rxs(i));
    %show(rxs(i));
end
%cover = SSs > rxSen;
%coverage(tx,'PropagationModel',pm,"SignalStrengths",-90:-5); 'lons' 'coverage' 'noncov'
%csvwrite('C:\materials\HKUST\PgYearOne\airCompProj\minitrain.csv','lats',0,0);
%csvwrite('C:\materials\HKUST\PgYearOne\airCompProj\minitrain.csv','lons',0,1);
%csvwrite('C:\materials\HKUST\PgYearOne\airCompProj\minitrain.csv','coverage',0,2);
%csvwrite('C:\materials\HKUST\PgYearOne\airCompProj\minitrain.csv','noncov',0,3);
%csvwrite('C:\materials\HKUST\PgYearOne\airCompProj\train.csv',[diff_Lat diff_Lon cover ~cover],1,0);
csvwrite('C:\materials\HKUST\PgYearOne\airCompProj\withHeight\test.csv',[diff_Lat diff_Lon eles SSs],1,0);