

#adapted from combinedPL files to plot the first and last file of a set of files

#%% set up details of experiment and process data into a single dataframe
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/ Hoke Effect/PL_data_300720/MAPbi25Br75/1hour' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))

chem = 'MAPbI25Br75'
exp = '1 hour'
laser = '488nm'
power = '5mW'
OD = '1p5'
integration = '10ms'
first_background = True #Use first frame as background

exp_seq = [[120,60],[58*60,58]] #measurements in the format [[time1, frames1], [time2, frames2]]
lim1 = 550 #high energy cutoff
lim2 = 900 #low energy cutoff

#%% initialize timing and determine delays
# TODO this cell should be a function

time_1 = all_files[0].split(" ")[6].split("_") 
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
#plot set time files
plt.figure(num = 1, figsize=(8,6))
fig1,ax1 = plt.subplots()
ax1.set_xlabel('Wavelength [nm]',size=14) #Define x-axis label
ax1.set_ylabel('Normalized Intensity',size=14)#Define y-axis label
ax1.set_xlim([525,900])
evenly_spaced_interval = np.linspace(0, 1, 5)
colors = [cm.rainbow(x) for x in evenly_spaced_interval]
#normalized
plt.plot(wave,PL_back[:,2]/max(PL_back[:,2]), color =  colors[0],label='t = 0')
plt.plot(wave,PL_back[:,29]/max(PL_back[:,29]), color = colors[1],label='t = 1 min')
plt.plot(wave,PL_back[:,59]/max(PL_back[:,59]), color = colors[2],label='t = 2 min')
plt.plot(wave,PL_back[:,-30]/max(PL_back[:,-30]), color = colors[3],label='t = 30 min')
plt.plot(wave,PL_back[:,-1]/max(PL_back[:,-1]), color = colors[4],label='t = 60 min')
#not normalized version
#plt.plot(wave,PL_back[:,1], color =  colors[0],label='t = 0')
#plt.plot(wave,PL_back[:,29], color = colors[1],label='t = 1 min')
#plt.plot(wave,PL_back[:,59], color = colors[2],label='t = 2 min')
#plt.plot(wave,PL_back[:,-30], color = colors[3],label='t = 30 min')
#plt.plot(wave,PL_back[:,-1], color = colors[4],label='t = 60 min')
#add bromine and iodine maximum points
#plt.plot(float(maxbr_wave),1, 'o', color='blue', label = 'Br center')
#plt.plot(float(maxi_wave),1, 'o', color='black', label = 'I center')
plt.axvline(x=float(maxbr_wave), color='blue', linestyle='--',label = 'Br center')
plt.axvline(x=float(maxi_wave), color='black', linestyle='--',label = 'I center')
plt.legend(loc="upper right")
#plt.title('First and last PL files over 1 hour for'+ chem)






