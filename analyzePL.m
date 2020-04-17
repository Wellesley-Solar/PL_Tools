% analyzePL_onecsv.m by Jill (04/01/20)
%File for processing a folder of csv files using the new PL set up
%Adapted from the same type of file from SSRL 2020 trip

% analyzePl.m by Jill (2/20/20)
%File for processing a folder of csv's using the new PL set up
%Adapted from the same type of file from SSRL 2020 trip
%does not yet include offsets

myDir = uigetdir;
files= dir(fullfile(myDir,'*.csv'));
figure()
hold on
%process and plot data
for k=6:length(files)
    fname = files(k).name
    fullfname = fullfile(myDir, fname);
    dat = importdata(fullfname);    %Imports csv
    wavelen = dat.data(:,3);  %Process
    intens = dat.data(:,6);
    plot(wavelen, intens);
end
hold off
xlabel('Wavelength (nm)')
ylabel('Intensity');
[~,name,~]=fileparts(myDir)
title(name);
