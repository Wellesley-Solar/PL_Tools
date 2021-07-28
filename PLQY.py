# Made by Prof. Belisle on 07/27/21 to calculate PLQY using multiple measurements taken with labview
#%% Import necessary packages and functions
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def find_nearest(array, value):
    #array is a 1D vector of wavelengths
    #value is the specific wavelength for which want the index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def trim_data(x, data, limit1, limit2):
    #x is a 1D array of two theta or q values
    #data is an array of x-ray intensities
    #limit1 and limit2 are what you'd like to trime your data to 
    test = np.array(x)
    set1 = find_nearest(x,limit1)
    set2 = find_nearest(x,limit2)
    return x[set1:set2], data[set1:set2]

def back_subtract(x, data, length):
    #x is a 1D array of wavelengths
    #data is an array of x-ray intensities
    #length is the number of values on the edges of the data you want to use to create a linear background 
    x_linear = np.hstack((x[0:length], x[-length:-1])) #I'm taking the starting and ending values
    data_linear = np.hstack((data[0:length], data[-length:-1])) #We'll use these to fit a straight line
    slope, intercept = np.polyfit(x_linear, data_linear, 1) #Do linear fit
    back = slope*x+intercept 
    data_correct=(data-back)
    return data_correct

#%% Import Data
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/PLQE Calibration/210722/Rhodamine/' #identify path where spectra are stored
all_files = sorted(glob.glob(path + "/*.csv"))
file_type = ['empty', 'Laser_off', 'PL_off', 'Laser_on', 'PL_on'] #specify order of experiments
integration_time = [10, 10, 500, 10, 500] #integration time of each experiment in ms

data = []

for i in range(len(all_files)):
    temp = pd.read_csv(all_files[i], index_col=None, header=0) #read file
    initial_I = temp["Intensity"]
    correct_I = np.array(back_subtract(temp["Wavelength"],temp["Intensity"],10))
    data.append([np.array(temp["Wavelength"]),correct_I*np.max(integration_time)/integration_time[i]]) #read in data and store it
# %% Do calcuations of peak areas
Laser_limits = [480, 500] #set limits over which you want to integrate the laser signal
PL_limits = [550,800] #set limits over which you want to integrate the PL signal

empty = trim_data(data[file_type.index('empty')][0], data[file_type.index('empty')][1], Laser_limits[0], Laser_limits[1])
empty_counts = np.sum(empty[1])

L_off = trim_data(data[file_type.index('Laser_off')][0], data[file_type.index('Laser_off')][1], Laser_limits[0], Laser_limits[1])
L_off_counts = np.sum(L_off[1])

PL_off = trim_data(data[file_type.index('PL_off')][0], data[file_type.index('PL_off')][1], PL_limits[0], PL_limits[1])
PL_off_counts = np.sum(PL_off[1])

L_on = trim_data(data[file_type.index('Laser_on')][0], data[file_type.index('Laser_on')][1], Laser_limits[0], Laser_limits[1])
L_on_counts = np.sum(L_on[1])

PL_on = trim_data(data[file_type.index('PL_on')][0], data[file_type.index('PL_on')][1], PL_limits[0], PL_limits[1])
PL_on_counts = np.sum(PL_on[1])

# %% Calculate PLQE According to Friend paper
A = 1-L_on_counts/L_off_counts
PLQY = (PL_on_counts-(1-A)*PL_off_counts)/(empty_counts*A)
print('PLQY', PLQY*100,'%')

# %% Plot emissions 
scale  = 100 #amount you want to scale PL by for it to be visible

fig1, ax1, = plt.subplots()
ax1.plot(empty[0], empty[1], 'k-', label = 'Empty')
ax1.plot(L_off[0], L_off[1], 'b-', label = 'Off Sample')
ax1.plot(PL_off[0], PL_off[1]*scale, 'b--')
ax1.plot(L_on[0], L_on[1], 'r-', label = 'On Sample')
ax1.plot(PL_on[0], PL_on[1]*scale, 'r--')
ax1.legend(loc='best')
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label

# %%
