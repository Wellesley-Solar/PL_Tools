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
from PLfunctions import sample_name, find_nearest, weighted_PL

path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/ Hoke Effect/PL_data_300720/MAPbI3/2minutes' # use your path
all_files = glob.glob(path + "/*.csv")
#%% set up details of experiment and process data into a single dataframe
chem = 'MAPbI3'
exp = '2min'
laser = '488nm'
power = '5mW'
OD = '1p5'

fulltime = 120 #total measument in seconds
integration = 10 #integration time in ms
interval = fulltime/len(all_files)
time = []

#pull wavelengths
df = pd.read_csv(all_files[0], index_col=None, header=0)
fullset = df[["Wavelength"]]

for x in range(len(all_files)):
    df = pd.read_csv(all_files[x], index_col=None, header=0) #read file
    current = round(x*interval,2) #assign time to file 
    temp = df.add_suffix(str(current)) #link time to file 
    fullset = pd.concat([fullset, temp[["Intensity"+str(current)]]], axis=1) #add intensity data to file
    time.append(current) #save time

fullset.to_csv(path+"/CombinedData/"+'ALL'+sample_name(chem,exp,laser,power,OD,str(integration)))

# %% Breaking down data for plotting full spects
fulldata = fullset.values
wave = fulldata[:,0]
PL = fulldata[:,1:]

# %% subtract background
ref = find_nearest(wave,900) #wavelength you will use as background
PL_back = PL
for file in range(len(fullset.columns)-1):
    PL_back[:,file] = PL[:,file] - np.mean(PL[(ref-10):(ref+10),file])
#%%
plt.figure(figsize=(8,6))
fig,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Intensity [a.u.]',size=14)#Define y-axis label
colors = [cm.gray(x) for x in range(len(fullset.columns)-1)]
for file in range(len(fullset.columns)-1):
    plt.plot(wave,PL_back[:,file], color = colors[file])

# %%Plot transient
cofm_PL = []
for file in range(len(fullset.columns)-1):
    cofm_PL.append(weighted_PL(wave,PL_back[:,file]))

fig,ax1 = plt.subplots()
ax1.set_xlabel('Time [s]',size=14) #Define x-axis label
ax1.set_ylabel('Weighted Average PL [nm]',size=14)#Define y-axis label
ax1.set_ylim([600,800])
plt.plot(time,cofm_PL, 'ko--', label=chem)
plt.legend(loc="lower right")#Put legend in upper left hand corner
# %%
