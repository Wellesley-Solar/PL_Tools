#a place for functions related to analysis of PL data
import pandas as pd
import numpy as np

def sample_name(chemistry,experiment,laser,power,OD,integration):
    return chemistry+'_'+experiment+'_'+laser+'_'+power+'_'+OD+'_'+integration+'.csv'


def find_nearest(array, value):
    #array is a 1D vector of wavelengths
    #value is the specific wavelength for which want the index
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def weighted_PL(wave,intensity):
    return np.sum(wave*intensity)/np.sum(intensity)