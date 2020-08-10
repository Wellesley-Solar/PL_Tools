#This program is meant to combine data taken over 
#subsequent experiments for single time series

#%% Import required packages
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PLfunctions import sample_name, find_nearest, weighted_PL, trim_data

#%% set up details of experiment and process data into a single dataframe
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/ Hoke Effect/PL_data_300720/MAPbI2Br/1hour' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))

chem = 'MAPbI2Br'
exp = '1 hour'
laser = '488nm'
power = '5mW'
OD = '1p5'
integration = '10ms'
first_background = True #Use first frame as background

exp_seq = [[120,60],[58*60,58]] #measurements in the format [[time1, frames1], [time2, frames2]]
lim1 = 550 #high energy cutoff
lim2 = 850 #low energy cutoff

#%% initialize timing and determine delays
# TODO this cell should be a function

time_1 = all_files[0].split(" ")[-2].split("_") 
start_time = [int(x) for x in time_1]
start_sec = start_time[0]*60**2+start_time[1]*60+start_time[2]
time_2 = all_files[exp_seq[0][1]].split(" ")[-2].split("_") 
restart_time = [int(x) for x in time_2]
restart_sec = restart_time[0]*60**2+restart_time[1]*60+restart_time[2]
delay = restart_sec-start_sec

#%%
# TODO should make it such that this can accept an arbitrary experimental sequence

f_times = []
df = pd.read_csv(all_files[0])
fullset = df[["Wavelength"]]

for f_name in all_files[0:exp_seq[0][1]]:

    f_idx = int(f_name.split("-")[-1].split(".")[0])
    f_time = round(f_idx * exp_seq[0][0]/exp_seq[0][1], 2)
    f_times.append(f_time)

    df = pd.read_csv(f_name, index_col=None, header=0) #read file
    temp = df.add_suffix(str(f_time)) #link time to file 
    fullset = pd.concat([fullset, temp[["Intensity"+str(f_time)]]], axis=1) #add intensity data to file

for f_name in all_files[exp_seq[0][1]:]:

    f_idx = int(f_name.split("-")[-1].split(".")[0])
    f_time = round(f_idx * exp_seq[1][0]/exp_seq[1][1], 2)+delay
    f_times.append(f_time)

    df = pd.read_csv(f_name, index_col=None, header=0) #read file
    temp = df.add_suffix(str(f_time)) #link time to file 
    fullset = pd.concat([fullset, temp[["Intensity"+str(f_time)]]], axis=1) #add intensity data to file


fullset.to_csv(path+"/CombinedData/"+'ALL'+sample_name(chem,exp,laser,power,OD,str(integration)))

# TODO may want to add a line that creates this folder if it does not exist
#  Breaking down data for plotting full specta
#%%
fulldata = fullset.values.copy()
wave = fulldata[:,0]
PL = fulldata[:,1:]
#%%
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

#%%
# Plot every file
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
# Plot transient
cofm_PL = []
for x in range(len(fullset.columns)-1):
    cofm_PL.append(weighted_PL(wave,PL_back[:,x]))
plt.figure(num = 2, figsize=(8,6))
fig2,ax2 = plt.subplots()
ax2.set_xlabel('Time [s]',size=14) #Define x-axis label
ax2.set_ylabel('Weighted Average PL [nm]',size=14)#Define y-axis label
ax2.set_ylim([650,850])
plt.plot(f_times,cofm_PL, 'ko--', label=chem)
plt.legend(loc="lower right")#Put legend in upper left hand corner
# %%
