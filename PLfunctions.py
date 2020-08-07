#a place for functions related to analysis of PL data
#%%
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def sample_name(chemistry,experiment,laser,power,OD,integration):
    return chemistry+'_'+experiment+'_'+laser+'_'+power+'_'+OD+'_'+integration+'.csv'

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
    set1 = find_nearest(x,limit1)
    set2 = find_nearest(x,limit2)
    return x[set1:set2], data[set1:set2,:]

def weighted_PL(wave,intensity):
    return np.sum(wave*intensity)/np.sum(intensity)

def weighted_PL_lim(wave,intensity,lim1,lim2):
    start = find_nearest(wave,lim1)
    end = find_nearest(wave,lim2)
    return np.sum(wave[lim1:lim2]*intensity[lim1:lim2])/np.sum(intensity[lim1:lim2])

# %%
