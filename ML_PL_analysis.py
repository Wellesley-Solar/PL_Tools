#This program is meant to combine data taken over 
#subsequent experiments for single time series
#%%
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PLfunctions import sample_name, find_nearest, weighted_PL, trim_data
from xrdfunctions import back_subtract

# %%    Create dictionary for storing analyzed results - to be run at the start of a new session
samplelist = []
results = {"Sample":"Results"}

#%%     Set up details of experiment and process data into a single dataframe
path = r'/Users/rbelisle/Desktop/4_1_21_PL/Data/chem50' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))
chem = 'chem33'
exp = '1 hour'
laser = '488nm'
power = '5mW'
OD = '1p5'
integration = '20ms'
first_background = True #Use first frame as background

exp_frames = [180, 6] #measurements in the format [[time1, frames1], [time2, frames2]]
lim1 = 550 #high energy cutoff
lim2 = 950 #low energy cutoff

#   Initialize timing and determine delays
#   TODO this cell should be a function
time_1 = all_files[0].split(" ")[1].split("_") 
start_time = [int(x) for x in time_1]
start_sec = start_time[0]*60**2+start_time[1]*60+start_time[2]
time_2 = all_files[exp_frames[0]].split(" ")[1].split("_") 
restart_time = [int(x) for x in time_2]
restart_sec = restart_time[0]*60**2+restart_time[1]*60+restart_time[2]
delay = restart_sec-start_sec

#   Pull start times from dataset
times = []
df = pd.read_csv(all_files[0])
fullset = df[["Wavelength"]]
time_zero = np.array((df[["Exposure started time stamp"]].iloc[0]))[0]

df = pd.read_csv(all_files[exp_frames[0]])
time_zero2 = np.array((df[["Exposure started time stamp"]].iloc[0]))[0]

for f_name in all_files[0:exp_frames[0]]:

    df = pd.read_csv(f_name, index_col=None, header=0) #read file
    current = np.array((df[["Exposure started time stamp"]].iloc[0]))[0] #assign time to file 
    exp_time = (current-time_zero)*10**(-6) #takes exposure start time and converts to seconds
    temp = df.add_suffix(str(exp_time)) #link time to file 
    fullset = pd.concat([fullset, temp[["Intensity"+str(exp_time)]]], axis=1) #add intensity data to file
    times.append(exp_time) #save time 

for f_name in all_files[exp_frames[0]:]:

    df = pd.read_csv(f_name, index_col=None, header=0) #read file
    current = np.array((df[["Exposure started time stamp"]].iloc[0]))[0] #assign time to file 
    exp_time = (current-time_zero2)*10**(-6) #takes exposure start time and converts to seconds
    temp = df.add_suffix(str(exp_time)) #link time to file 
    fullset = pd.concat([fullset, temp[["Intensity"+str(exp_time)]]], axis=1) #add intensity data to file
    times.append(exp_time+delay) #save time 

fullset.to_csv(path+'ALL'+sample_name(chem,exp,laser,power,OD,str(integration)))
# TODO may want to add a line that creates this folder if it does not exist

#%%     Breaking down data for plotting full specta
fulldata = fullset.values.copy()
wave = fulldata[:,0]
PL = fulldata[:,1:]
times = np.array(times)
laser_on = 2 #frame at which the laser turns on

#   Subtract background
if first_background:
    dark = laser_on - 1
    PL_back = PL
    wave_back, PL_back = trim_data(wave,PL_back,lim1,lim2)
    for x in range(len(fullset.columns)-1):
        PL_back[:,x] = PL_back[:,x] - PL_back[:,dark]
        PL_back[:,x] = back_subtract(wave_back, PL_back[:,x],10)
    PL_back = PL_back[:,laser_on:]
    time_s = times[laser_on:]-times[laser_on]
    
else: 
    PL_back = PL
    wave_back, PL_back = trim_data(wave,PL_back,lim1,lim2)
    for x in range(len(fullset.columns)-1):
        PL_back[:,x] = back_subtract(wave_back, PL_back[:,x],10)
    PL_back = PL_back[:,laser_on:]
    time_s = times[laser_on:]-times[laser_on]

#%%     Plot every file
plt.figure(num = 1, figsize=(8,6))
fig1,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label
ax1.set_xlim([lim1,lim2])
evenly_spaced_interval = np.linspace(0, 1, len(time_s))
# colors = [cm.gray(x) for x in range(len(fullset.columns)-1)]
colors = [cm.jet(x) for x in evenly_spaced_interval]
for x, color in enumerate(colors):
    plt.plot(wave_back,PL_back[:,x], color = color)
# TODO PLOT needs to iteratre through small color range 

# %%    Plot transient
cofm_PL = []
for x in range(len(time_s)):
    cofm_PL.append(weighted_PL(wave_back,PL_back[:,x]))
plt.figure(num = 2, figsize=(8,6))
fig2,ax2 = plt.subplots()
ax2.set_xlabel('Time [s]',size=14) #Define x-axis label
ax2.set_ylabel('Weighted Average PL [nm]',size=14)#Define y-axis label
#ax2.set_ylim([650,800])
ax2.set_xlim([0,500])
plt.plot(time_s,cofm_PL, 'ko--', label=chem)
plt.legend(loc="lower right")#Put legend in upper left hand corner


# %%    Keep data
samplelist.append(chem) 
datatostore = [time_s,  cofm_PL]
results[chem] = datatostore
# %%
samplelist_sub = samplelist[3:-1]
plt.figure(num = 2, figsize=(8,6))
fig2,ax2 = plt.subplots()
for each in samplelist_sub:
    plt.plot(results[each][0],results[each][1], '.--', label = each)

plt.legend(loc="upper center", ncol=4)#Put legend in upper left hand corner 
ax2.set_xlabel('Time [s]',size=14) #Define x-axis label
ax2.set_ylim([670,760])
ax2.set_xlim([0,1750])
ax2.set_ylabel('Weighted Average PL [nm]',size=14)#Define y-axis label
# %%
