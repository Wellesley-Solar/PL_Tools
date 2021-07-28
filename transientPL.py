#should be expanded into a juptyer notbook that goes over:
#importing data
#correcing for integration
#pulling time
#plot
#fit
#plot dynamic behavior
#%%
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PLfunctions import sample_name, find_nearest, weighted_PL, trim_data
#%% set up details of experiment and process data into a single dataframe
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/ Hoke Effect/PL/PL_data_300720/MAPbI5Br5/2min' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))
chem = 'MAPbI5Br'
exp = '2min'
laser = '488nm'
power = '5mW'
OD = '1p5'
delay = 120 #time experiment is running before data 
lim1 = 550 #high energy cutoff
lim2 = 850 #low energy cutoff
first_background = True #if first file should be used as background

fulltime = 85*60 #total measument in seconds
integration = 10 #integration time in ms
interval = fulltime/len(all_files)
time = []

#pull wavelengths
df = pd.read_csv(all_files[0])
fullset = df[["Wavelength"]]


for x in range(len(all_files)):
    df = pd.read_csv(all_files[x], index_col=None, header=0) #read file
    current = round(x*interval,2) #assign time to file 
    temp = df.add_suffix(str(current)) #link time to file 
    fullset = pd.concat([fullset, temp[["Intensity"+str(current)]]], axis=1) #add intensity data to file
    time.append(current+delay) #save time

fullset.to_csv(path+"/CombinedData/"+'ALL'+sample_name(chem,exp,laser,power,OD,str(integration)))
#may want to add a line that creates this folder if it does not exist
#  Breaking down data for plotting full specta
fulldata = fullset.values.copy()
wave = fulldata[:,0]
PL = fulldata[:,1:]

# subtract background
PL_back = PL
if first_background:
    PL_back = PL
    ref = find_nearest(wave,900) #wavelength you will use as background

    for x in range(len(fullset.columns)-1):
        PL_back[:,x] = PL[:,x] - np.mean(PL[(ref-10):(ref+10),x]) 
    for x in range(len(fullset.columns)-1):
        PL_back[:,x] = PL_back[:,x] - PL_back[:,0]
    wave, PL_back = trim_data(wave,PL_back,lim1,lim2)

else: 
    # subtract background
    ref = find_nearest(wave,900) #wavelength you will use as background
    PL_back = PL

    for x in range(len(fullset.columns)-1):
        PL_back[:,x] = PL[:,x] - np.mean(PL[(ref-10):(ref+10),x]) 

    wave, PL_back = trim_data(wave,PL_back,lim1,lim2)

#%% Plot every file
plt.figure(num = 1, figsize=(8,6))
fig1,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label
ax1.set_xlim([lim1,lim2])
evenly_spaced_interval = np.linspace(0, 1, len(fullset.columns)-1)
# colors = [cm.gray(x) for x in range(len(fullset.columns)-1)]
colors = [cm.rainbow(x) for x in evenly_spaced_interval]
for x, color in enumerate(colors):
    plt.plot(wave,PL_back[:,x], color = color)
# TODO PLOT needs to iteratre through small color range 
# %%Plot transient
cofm_PL = []
for x in range(len(fullset.columns)-1):
    cofm_PL.append(weighted_PL(wave,PL_back[:,x]))
plt.figure(num = 2, figsize=(8,6))
fig2,ax2 = plt.subplots()
ax2.set_xlabel('Time [s]',size=14) #Define x-axis label
ax2.set_ylabel('Weighted Average PL [nm]',size=14)#Define y-axis label
ax2.set_ylim([650,800])
plt.plot(time,cofm_PL, 'ko--', label=chem)
plt.legend(loc="lower right")#Put legend in upper left hand corner
# %%