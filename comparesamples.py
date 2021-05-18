
#import multiple samples
#plot PL at initial (may need some help)
#plot PL at end
#plot transients
#%% Import required packages
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
from PLfunctions import sample_name, find_nearest, weighted_PL, trim_data, exp_fit

#%% identify location of relevant files
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/ Hoke Effect/PL/PL_data_300720/All chemistries' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))

# %% manually assign all data to chemistry
Br75 = np.array(pd.read_csv(all_files[0],index_col=None, header=0))
Br0 = np.array(pd.read_csv(all_files[2],index_col=None, header=0))
Br33 = np.array(pd.read_csv(all_files[1],index_col=None, header=0))
Br50 = np.array(pd.read_csv(all_files[3],index_col=None, header=0))
Br67 = np.array(pd.read_csv(all_files[4],index_col=None, header=0))
Br90 = np.array(pd.read_csv(all_files[-1],index_col=None, header=0))
#%% trim data
lim1 = 560
lim2 = 875
wave_75, Br75_trim = trim_data(Br75[:,1],Br75[:,2:],lim1,lim2)
wave_50, Br50_trim = trim_data(Br50[:,1],Br50[:,2:],lim1,lim2)
wave_33, Br33_trim = trim_data(Br33[:,1],Br33[:,2:],lim1,lim2)
wave_0, Br0_trim = trim_data(Br0[:,1],Br0[:,2:],lim1,lim2)
wave_67, Br67_trim = trim_data(Br67[:,1],Br67[:,2:],lim1,lim2)
wave_90, Br90_trim = trim_data(Br90[:,1],Br90[:,2:],lim1,lim2)
# %% Make plots of initial PL (a.k.a. time zero)
plt.figure(num = 1, figsize=(8,6))
fig1,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label
#ax1.set_ylim([0,1000])
ax1.set_xlim([550,900])
plt.plot(wave_0,(Br0_trim[:,0]-Br0_trim[1,0])/max((Br0_trim[:,0]-Br0_trim[1,0])), 'k-', label='x=0')
plt.plot(wave_33,(Br33_trim[:,1]-Br33_trim[:,0])/max(Br33_trim[:,1]-Br33_trim[:,0]), 'b-', label='x=0.33')
plt.plot(wave_50,(Br50_trim[:,1]-Br50_trim[:,0])/max(Br50_trim[:,1]-Br50_trim[:,0]), 'r-',label='x=0.50')
plt.plot(wave_67,(Br67_trim[:,1]-Br67_trim[:,0])/max(Br67_trim[:,1]-Br67_trim[:,0]), 'g-', label='x=0.67')
plt.plot(wave_75,(Br75_trim[:,2]-Br75_trim[:,0])/max(Br75_trim[:,2]-Br75_trim[:,0]), 'm-', label='x=0.75')
#plt.plot(wave_90,(Br90_trim[:,3]-Br90_trim[:,0])/max(Br90_trim[:,2]-Br90_trim[:,0]), 'y-')
plt.legend(loc="upper right")#Put legend in upper left hand corner
#TODO look to see if 90% high energy PL laser or PL
#TODO determine peak PL wavelength for doing initial bandgap PL
#fig1.savefig(path+"/10ms.png")

#%% Determine initial wavelength
initial_PL = []
initial_PL.append(wave_0[np.abs(Br0_trim[:,1] - max(Br0_trim[:,1])).argmin()])
initial_PL.append(wave_33[np.abs(Br33_trim[:,1] - max(Br33_trim[:,1])).argmin()])
initial_PL.append(wave_50[np.abs(Br50_trim[:,1] - max(Br50_trim[:,1])).argmin()])
initial_PL.append(wave_67[np.abs(Br67_trim[:,1] - max(Br67_trim[:,1])).argmin()])
initial_PL.append(wave_75[np.abs(Br75_trim[:,2] - max(Br75_trim[:,2])).argmin()])
initial_PL= [1240/x for x in initial_PL]
chem = [0, .33, .50, .67, .75]
# %% Make plots of at 2 minutes
plt.figure(num = 1, figsize=(8,6))
fig1,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label
#ax1.set_ylim([0,1000])
ax1.set_xlim([575,900])
plt.plot(wave_0,(Br0_trim[:,59]-Br0_trim[1,0])/max((Br0_trim[:,59]-Br0_trim[1,0])), 'k-', label='x=0')
plt.plot(wave_33,(Br33_trim[:,59]-Br33_trim[:,0])/max(Br33_trim[:,59]-Br33_trim[:,0]), 'b-', label='x=0.33')
plt.plot(wave_50,(Br50_trim[:,59]-Br50_trim[:,0])/max(Br50_trim[:,59]-Br50_trim[:,0]), 'r-',label='x=0.50')
plt.plot(wave_67,(Br67_trim[:,59]-Br67_trim[:,0])/max(Br67_trim[:,59]-Br67_trim[:,0]), 'g-', label='x=0.67')
plt.plot(wave_75,(Br75_trim[:,59]-Br75_trim[:,0])/max(Br75_trim[:,59]-Br75_trim[:,0]), 'm-', label='x=0.75')
#plt.plot(wave_90,(Br90_trim[:,3]-Br90_trim[:,0])/max(Br90_trim[:,2]-Br90_trim[:,0]), 'y-')
plt.legend(loc="upper right")#Put legend in upper left hand corner
#TODO look to see if 90% high energy PL laser or PL
#TODO repeat with doing faster data collection, can look at first second
fig1.savefig(path+"/2min.png")
# %%
plt.figure(num = 1, figsize=(8,6))
fig1,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label
ax1.set_ylim([0,1.1])
ax1.set_xlim([565,900])
plt.plot(Br0[:,1],(Br0[:,-1]-Br0[10,2])/max(Br0[:,-1]-Br0[10,2]), 'k-',label='x=0')
plt.plot(Br33[:,1],(Br33[:,-1]-Br33[:,2])/max(Br33[:,-1]-Br33[:,2]), 'b-',label='x=0.33')
plt.plot(Br50[:,1],(Br50[:,-1]-Br50[:,2])/max(Br50[:,-1]-Br50[:,2]), 'r-',label='x=0.50')
plt.plot(Br67[:,1],(Br67[:,-1]-Br67[:,2])/max(Br67[:,-1]-Br67[:,2]), 'g-',label='x=0.67')
plt.plot(Br75[:,1],(Br75[:,-1]-Br75[:,2])/max(Br75[:,-1]-Br75[:,2]), 'm-', label='x=0.75')
plt.plot(Br90[:,1],(Br90[:,-1]-Br90[:,3])/max(Br90[:,-1]-Br90[:,3]), 'y-',label='x=0.90')
plt.legend(loc="upper left")#Put legend in upper left hand corner
fig1.savefig(path+"/1hour.png")


# %% Final Peak in eV
final_PL = []
popt,pcov = curve_fit(gaussian, wave_0, Br0_trim[:,-1], p0 = [1000, 800, 10], maxfev=8000)
plt.plot(wave_0, Br0_trim[:,-1], 'ko')
plt.plot(wave_0,gaussian(wave_0,*popt),'k--')
p0 = popt
final_PL.append(popt[1])
popt,pcov = curve_fit(gaussian, wave_33, Br33_trim[:,-1], p0, maxfev=8000)
plt.plot(wave_33, Br33_trim[:,-1], 'ko')
plt.plot(wave_33,gaussian(wave_33,*popt),'k--')
p0 = popt
final_PL.append(popt[1])
popt,pcov = curve_fit(gaussian, wave_50, Br50_trim[:,-1], p0, maxfev=8000)
plt.plot(wave_50, Br50_trim[:,-1], 'ko')
plt.plot(wave_50,gaussian(wave_50,*popt),'k--')
p0 = popt
final_PL.append(popt[1])
popt,pcov = curve_fit(gaussian, wave_67, Br67_trim[:,-1], p0, maxfev=8000)
plt.plot(wave_67, Br67_trim[:,-1], 'ko')
plt.plot(wave_67,gaussian(wave_67,*popt),'k--')
p0 = popt
final_PL.append(popt[1])
popt,pcov = curve_fit(gaussian, wave_75, Br75_trim[:,-1], p0, maxfev=8000)
plt.plot(wave_75, Br75_trim[:,-1], 'ko')
plt.plot(wave_75,gaussian(wave_75,*popt),'k--')
p0 = popt
final_PL.append(popt[1])
final_PLev=[1240/x for x in final_PL]
popt,pcov = curve_fit(gaussian, wave_90, Br90_trim[:,-1], p0, maxfev=8000)
plt.plot(wave_90, Br90_trim[:,-1], 'ko')
plt.plot(wave_90,gaussian(wave_90,*popt),'k--')
p0 = popt
final_PL.append(popt[1])
final_PLev=[1240/x for x in final_PL]
# %%
