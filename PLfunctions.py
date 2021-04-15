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
    return np.sum(wave[start:end]*intensity[start:end])/np.sum(intensity[start:end])

def exp_fit(x,a,b,c):
    x = np.array(x)
    return -1*a*np.exp(x*-1*b)+c

def back_subtract(x, data, length):
    #x is a 1D array of two theta or q values
    #data is an array of x-ray intensities
    #length is the number of values on the edges of the data you want to use to create a linear background 
    x_linear = np.hstack((x[0:length], x[-length:-1])) #I'm taking the starting and ending values
    data_linear = np.hstack((data[0:length], data[-length:-1])) #We'll use these to fit a straight line
    slope, intercept = np.polyfit(x_linear, data_linear, 1) #Do linear fit
    back = slope*x+intercept 
    data_correct=(data-back)
    return data_correct


# %%
