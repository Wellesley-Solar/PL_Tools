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
from PLfunctions import back_subtract, sample_name, find_nearest, weighted_PL, trim_data
#%% set up details of experiment and process data into a single dataframe
path = r'/Volumes/GoogleDrive/Shared drives/Wellesley Solar/Current Projects/Fall 2021 New Triple Halide Perovskite Search/211223_round2_PL' # use your path
all_files = sorted(glob.glob(path + "/*.csv"))
exp = '4hrs'
laser = '488nm'
power = '5mW'
OD = '1p5'
lim1 = 600 #high energy cutoff
lim2 = 850 #low energy cutoff

#%%
#open file
for each in all_files:
    print(chem)
    df = pd.read_csv(each)
    chem = each.split('_')[-1].split('.')[0]

    #Process data
    cofm_PL = []
    time = []
    fig1,ax1 = plt.subplots()
    init_time = np.min(df['Exposure started time stamp'])

    first = df[df['Frame'] == 1]['Intensity'] #read file
    last = df[df['Frame'] == np.max(df['Frame'])]['Intensity']
    #%%

    for x in range(1,np.max(df['Frame'])):
        current = df[df['Frame'] == x] #read file
        wave = current['Wavelength']
        intensity = current['Intensity'] 
        wave_cut, PL_cut = trim_data(wave,intensity,lim1,lim2)
        PL_back = PL_cut - np.mean(PL_cut[0:10])
        ax1.plot(wave_cut,PL_back)
        cofm_PL.append(weighted_PL(wave_cut,PL_back))
        time.append((current['Exposure ended time stamp']+current['Exposure started time stamp'])/2)
    ax1.set_xlabel("Wavelength [nm]")
    ax1.set_ylabel("Intensity [nm]")
    ax1.set_title(chem)
    
    #fig1.savefig(path+"/"+chem+".png")

    true_time = (time-init_time)/(10**6)
    fig2,ax2 = plt.subplots()
    ax2.plot(true_time, cofm_PL, 'k--')
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Center of Mass Wavelength [nm]")
    ax2.set_title(chem)
    #fig2.savefig(path+"/"+chem+".png")

    fig3,ax3 = plt.subplots()
    ax3.plot(wave, first, 'b-', label = "first frame")
    ax3.plot(wave,last, 'r-', label = "last frame")
    ax3.set_xlabel("Wavelength [nm]")
    ax3.set_ylabel("Intensity [nm]")
    ax3.set_xlim([600,850])
    ax3.set_title(chem)
    ax3.legend(loc="upper right")

    plt.show()
    plt.close('all')

# %%
