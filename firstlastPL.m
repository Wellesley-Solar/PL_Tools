%firstlastPL by Jill (04/01/20)
%File for processing a folder of csv files using the new PL set up
%Plots only the first and last files

myDir = uigetdir;
files= dir(fullfile(myDir,'*.csv'));

figure()
hold on
%process and plot data
for k=3:length(files)-4:length(files)
    fname = files(k).name
    fullfname = fullfile(myDir, fname);
    dat = importdata(fullfname);    %Imports csv
    wavelen = dat.data(:,3);  %Process
    intens = dat.data(:,6);
    plot(wavelen, intens, 'linewidth', 2);
end
hold off
xlabel('Wavelength (nm)')
ylabel('Intensity');
[~,name,~]=fileparts(myDir)
title(name);
legend('first','last')
