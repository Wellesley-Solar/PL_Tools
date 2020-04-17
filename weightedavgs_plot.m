% weightedavg_plot.m by Jill (4/7/20)
%File for processing a folder of csv's using the new PL set up
%Adapted from the same type of file from SSRL 2020 trip
%does not yet include offsets
%includes calculation of weighted average that goes to an array by frame

myDir = uigetdir;
files= dir(fullfile(myDir,'*.csv'));
temps = []
weightedavgs = []

%hold on
%process and plot data
for k=6:length(files)
    fname = files(k).name
    fullfname = fullfile(myDir, fname);
    dat = importdata(fullfname);    %Imports csv
    wavelen = dat.data(:,3);  %Process
    intens = dat.data(:,6);
    wint = wavelen.*intens;
    tot = sum(intens);
    weightavg = sum(wint)/tot % calculates weighted avg for entire frame
    weightedavgs = [weightedavgs weightavg];
   % plot(wavelen, intens);
end
%hold off
%xlabel('Wavelength (nm)')
%ylabel('Intensity');
[~,name,~]=fileparts(myDir)
%title(name);

%plot weighted average across time series
figure()
plot(weightedavgs);
xlabel('frame')
ylabel('Intensity weighted avg');
title('intensity weighted averages');
[~,name,~]=fileparts(myDir)
title(name);