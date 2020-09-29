# Import required packages
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
# from PLfunctions import sample_name, find_nearest, weighted_PL, trim_data, exp_fit
import math
import scipy.optimize 
from lmfit import models
from numpy import random
from scipy import signal

class PLspec:
    """
    PLspec accepts the path to a .csv file containing spectrometer output with two columns 
    named "Wavelength" and "Intensity." The wavelength series is saved in PLspec.W, the 
    intensity series in PLspec.I, the photon energy series corresponding to the wavelength 
    in PLspec.I. The intensity-weighted wavelength/photon energy over time is also calculated
    upon initializing and saved in PLspec.IweightedW or PLspec.IweightedE.
    
    """
    def __init__(self, file):
        """Initiates the PLspec class by reading a spectrum csv into a dataframe
        
        Args:
            file: path to a .csv file containing spectrometer output with at least two columns that
            are named "Wavelength" and "Intensity." Intensity weight averages for wavelength and 
            photon energy are made available

        Returns:
            None
        
        """
        self.c = 2.99792458e8 * 1e9 #
        self.h = 4.1357e-15 #eV*s
        self.file = file
        self.og = pd.read_csv(self.file, index_col=None, header=0)
        self.df = self.og
        self.formatData()
        
    def formatData(self):
        """Perform basic intensity weighted average calculations and save data to appropriate attributes
        
        Args:
            None
        
        Returns:
            None
        
        """
        self.W = self.df["Wavelength"]
        self.I = self.df["Intensity"]
        self.E = self.c/self.W*self.h
        self.IweightedW = sum(self.I*self.W)/sum(self.I)
        self.IweightedE = sum(self.I*self.E)/sum(self.I)
        
    def plotW(self):
        """Plot the spectrum in the format of intensity versus wavelength
        
        Args:
            None
        
        Returns:
            None
        
        """
        fig, ax = plt.subplots()
        ax.scatter(self.W, self.I,s=4)
        ax.set_xlabel("Wavelength(nm)")
        ax.set_ylabel("Intensity(Counts)")
        ax.set_title(self.file, {'fontsize': "small",})
        
    
    def plotE(self):
        """Plot the spectrum in the format of intensity versus photon energy
        
        Args:
            None
        
        Returns:
            None
        
        """
        fig, ax = plt.subplots()
        ax.scatter(self.E, self.I,s=4)
        ax.set_xlabel("Photon Energy(eV)")
        ax.set_ylabel("Intensity(Counts)")
        ax.set_title(self.file, {'fontsize': "small",})
        
        
    def retore(self):
        """Remove the cutoff for the spectrum data
        
        Args:
            None
        
        Returns:
            None
        
        """
        self.df = self.og 
        self.formatData()

        
    def narrow(self, Wmin=550, Wmax=1000):
        """Enact the cutoff for the spectrum data
        
        Args:
            Wmin: optional minimum wavelength cutoff to focus on a narrower range of data, won't be
                implmented until PLspec.narrow() is called
            Wmax: optional maximum wavelength cutoff to focu on a narrower range of data, won't be
                implmented until PLspec.narrow() is called
                
        Returns:
            None
        
        """
        self.Wmin = Wmin
        self.Wmax = Wmax
        self.minCond = self.df["Wavelength"]>self.Wmin
        self.maxCond = self.df["Wavelength"]<self.Wmax
        self.df = self.df[self.minCond & self.maxCond]
        self.formatData()
        
    
    def gaussian(self, x, mu, sigma, height):
        """Definition of a single gaussian function
        
        Args:
            x: input data over which gaussian values are calculated
            mu: the center of the gaussian 
            sigma: the standard deviation of the gaussian
            height: the scaling of the gaussian
                
        Returns:
            None
        """
        return height*np.exp(-(x - mu)**2/(2*sigma**2))


class PLevol:
    def __init__(self, folder):
        """Initiates the PLevol class by converting full spectrum csv into individual PLspec class.
        A series of intensity weighted wavelength over time and another series of intensity weighted
        photon energy over time is calculated.
        
        Args:
            folder: the path to folder/directory in which spectrometer files are stored
        
        Returns:
            None
        
        """
        self.og = [PLspec(file) for file in sorted(glob.glob(folder+"/*.csv"))]
        self.PLs = self.og
        self.Wavgseries = [spec.IweightedW for spec in self.PLs]
        self.Eavgseries = [each.IweightedE for each in self.PLs]
        
    def restore(self):
        """Remove the cutoff for the spectrum data
        
        Args:
            None
        
        Returns:
            None
        
        """
        self.PLs = self.og 
        
    def narrow(self, Wmin=550, Wmax=1000):
        """Enact the cutoff for the spectrum data
        
        Args:
            file: path to a .csv file containing spectrometer output with at least two columns that
            are named "Wavelength" and "Intensity"
            Wmin: optional minimum wavelength cutoff to focus on a narrower range of data, won't be
            implmented until PLspec.narrow() is called
            Wmax: optional maximum wavelength cutoff to focu on a narrower range of data, won't be
            implmented until PLspec.narrow() is called
        
        Returns:
            None
        
        """
        [PL.narrow(Wmin, Wmax) for PL in self.og]
        self.Wavgseries = [each.IweightedW for each in self.PLs]
        self.Eavgseries = [each.IweightedE for each in self.PLs]

# Some of the following peak fitting code was originally written by Chris Ostrouchov
# You can find them here: https://chrisostrouchov.com/post/peak_fit_xrd_python/
# Qingmu Deng rewrote most of the original code for PL Fitting for the Belis Lab at Wellesley College
class Fitter():
    def __init__(self):
        self.output_series=[]
        self.first_pass=True


    def basicparam(self, spec):
        x = spec['x']
        y = spec['y']
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min
        y_max = np.max(y)
        return None, None, x, y, x_min, x_max, x_range, y_max


    def spec_gen(self, x, y, peaks, model='GaussianModel'):
        return {
            'x': x,
            'y': y,
            'model': [{'type': model} for i in range(len(peaks))]
            }

    def common_param_set(self, mdl, peak, error, y_max):
        """ Specify bounds for relevant model parameters
        Args:
            mdl: a lmfit model object
            peak: a list of [peak_x_value, peak_y_value]
            error: freedom a peak's center is given to the left and right in nm
            y_max: maximum y of the spectrum
        
        Returns:
            None
        """
        mdl.set_param_hint('sigma', min=5, max=50)
        mdl.set_param_hint('center', min=peak[0]-error, max=peak[0]+error)
        mdl.set_param_hint('height', min=1e-6, max=1.1*y_max)
        mdl.set_param_hint('amplitude', min=1e-6)
        
    def model_params_update(self, comp_model, model, params, model_params):
        """ Incorporate individual model and model parameters into composite models
        Args:
            comp_model: a lmfit composite model object that is the linear combination of individual model
            model: an individual lmfit model to be added to the composite model
            params: parameters for the composite model
            model_params: parameters of the single model to be added to the composite model parameters
        
        Returns:
            comp_model: a composite model combined with the additional model
            params: composite model parameters incorporated with the additional model parameters
        """
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if comp_model is None:
            comp_model = model
        else:
            comp_model = comp_model + model
        return comp_model, params


    def peaks2spec(self, x, y, peak_widths=(10,200)):
        """Generate an initial model from CWT peak fitting
        Calls my_modelgen()

        Args:
            x: spectrum x/wavelength data series
            y: spectrum y/intensity data series
            peak_widths: allowable peak width for CWT search
        
        Returns:
            peak_indicies: peak indices identified by CWT
            composite_model: fully assembled composite model for fitting
            model_params: fully assembled composite model parameters for fitting
        """
        peak_indicies = signal.find_peaks_cwt(y, peak_widths)
    #     print(peak_index)
        temp=[]
        # print(x)
        # print(peak_indicies)
        for peak_index in peak_indicies:
            if x[peak_index] > 600.0 and x[peak_index] < 800:
                temp.append(peak_index)
        peak_indicies=np.array(temp)
        spec = self.spec_gen(x, y, peak_indicies)
        
        composite_model, model_params = self.my_modelgen(spec, peak_indicies, peak_widths)

        return peak_indicies, composite_model, model_params
    

    def my_modelgen(self, spec, peak_indicies, peak_widths, error=10):
        """ Generate an initial composite model and model parameters based on CWT peak detection
        Args:
            spec: initial model specification variable, see spec_gen()
            peak_indicies: peak indices identified by CWT
            peak_widths: allowable peak width for CWT search
            error: freedom a peak's center is given to the left and right in nm
        
        Returns:
            
            composite_model: fully assembled composite model for fitting
            model_params: fully assembled composite model parameters for fitting
        """
        # Initiate and extract basic parameters from the specifications
        composite_model, params, x, y, x_min, x_max, x_range, y_max = self.basicparam(spec)
        
        # For each model specified in the specification, set their parameters bounds 
        # and add them together to form a composite model
        for (i, basis_func), peak_index in zip(enumerate(spec['model']), peak_indicies):
            prefix = f'm{i}_'
            model = getattr(models, basis_func['type'])(prefix=prefix)
            if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel','PseudoVoigtModel']: # for now VoigtModel has gamma constrained to sigma
                model.set_param_hint('sigma', min=5, max=50)#x_range)
                model.set_param_hint('center', min=x[peak_index]-error, max=x[peak_index]+error)
                model.set_param_hint('height', min=1e-6, max=1.1*y_max)
                model.set_param_hint('amplitude', min=1e-6)
                # avoid using default parameter guess
                default_params = {
                    prefix+'center': x[peak_index],
                    prefix+'height': y[peak_index],
                    prefix+'sigma': 50#x_range / len(x) * np.min(peak_widths)
                }
            else:
                raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
            
            if 'help' in basis_func:  # allow override of settings in parameter
                for param, options in basis_func['help'].items():
                    model.set_param_hint(param, **options)
                                    
            # make model parameters based on the 
            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            # Add up each individual models and update their parameters accordingly
            composite_model, params = self.model_params_update(composite_model, model, params, model_params) 
                                    
        return composite_model, params
                                 
                                    
                                    
    def both2spec(self, x, y, peak_widths=(10,200)):
        """ Generate both a model from CWT peak fitting and a model from last peak fitting parameters
        Calls last_modelgen() and cwt_modelgen()    
        
        Args:
            x: spectrum x/wavelength data series
            y: spectrum y/intensity data series
            peak_widths: allowable peak width for CWT search
        
        Returns:
            peak_indicies: peak indices
            composite_model: fully assembled composite model for fitting
            model_params: fully assembled composite model parameters for fitting    
        """
        
        # Do a fresh extraction of peak position with CWT
        peak_indicies = signal.find_peaks_cwt(y, peak_widths)
        peaks_cwt=[]
        for peak_index in peak_indicies:
            if x[peak_index] > 600.0 and x[peak_index] < 800:
                peaks_cwt.append([x[peak_index],y[peak_index]])
        
        # Consider the peak positions from the last fitting routine as well
        last_peaks=[]
        if self.output_series != []:
            last_out=self.output_series[-1]
            for each in last_out.params:
                if each.find("center") != -1:
                    amp = last_out.params[each[:3]+'amplitude']
                    if amp.value < 40:
                        continue
                    last_peaks.append([last_out.params[each].value, last_out.params[each[:3]+'height'].value,\
                                    last_out.params[each[:3]+'sigma'].value])
        
        # Generate a first model based on CWT-extracted peaks
        pk_spec = self.spec_gen(x, y, peaks_cwt)
        pk_model, pk_params = self.cwt_modelgen(pk_spec, peaks_cwt, peak_widths, error=10)
        # Generate a second model based on last peakfitting parameters
        lt_spec = self.spec_gen(x, y, last_peaks)
        lt_model, lt_params = self.last_modelgen(pk_spec, last_peaks, peak_widths, error=10)
        
        # Get the CWT peak locations for comparison
        peaks_cwt=[each[0] for each in peaks_cwt]
        return peaks_cwt, pk_model, pk_params, lt_model, lt_params


    def last_modelgen(self, spec, peaks, peak_widths, error=10):
        """ Generate a composite model and model parameters based on last fitting parameters
        Args:
            spec: initial model specification variable, see spec_gen()
            peak_indicies: peak indices identified by CWT
            peak_widths: allowable peak width for CWT search
            error: freedom a peak's center is given to the left and right in nm
        
        Returns:
            composite_model: fully assembled composite model for fitting
            model_params: fully assembled composite model parameters for fitting
        """
        # Initiate and extract basic parameters from the specifications
        composite_model, params, x, y, x_min, x_max, x_range, y_max = self.basicparam(spec)
        
        # For each model specified in the specification, set their parameters bounds 
        # and add them together to form a composite model
        for (i, basis_func), peak in zip(enumerate(spec['model']), peaks):
            prefix = f'm{i}_'
            model = getattr(models, basis_func['type'])(prefix=prefix)
            if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel','PseudoVoigtModel']: # for now VoigtModel has gamma constrained to sigma

                # model is passed by reference here so no need to return anything
                self.common_param_set(model, peak,error, y_max)

                # avoid using default parameter guess
                default_params = {
                    prefix+'center': peak[0],
                    prefix+'height': peak[1],
                    prefix+'sigma': peak[2]#x_range / len(x) * np.min(peak_widths)#peak[2]
                }
            else:
                raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
            
            if 'help' in basis_func:  # allow override of settings in parameter
                for param, options in basis_func['help'].items():
                    model.set_param_hint(param, **options)
            
            # make model parameters based on the 
            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            # Add up each individual models and update their parameters accordingly
            composite_model, params = self.model_params_update(composite_model, model, params, model_params)  

        return composite_model, params
                                 

    def cwt_modelgen(self, spec, peaks, peak_widths, error=10):
        """ Generate a composite model and model parameters based on CWT peak detections
        Args:
            spec: initial model specification variable, see spec_gen()
            peak_indicies: peak indices identified by CWT
            peak_widths: allowable peak width for CWT search
            error: freedom a peak's center is given to the left and right in nm
        
        Returns:
            composite_model: fully assembled composite model for fitting
            model_params: fully assembled composite model parameters for fitting
        """
        # Initiate and extract basic parameters from the specifications
        composite_model, params, x, y, x_min, x_max, x_range, y_max = self.basicparam(spec)
        
        # For each model specified in the specification, set their parameters bounds 
        # and add them together to form a composite model                       
        for (i, basis_func), peak in zip(enumerate(spec['model']), peaks):
            prefix = f'm{i}_'
            model = getattr(models, basis_func['type'])(prefix=prefix)
            if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel','PseudoVoigtModel']: # for now VoigtModel has gamma constrained to sigma
                # model is passed by reference here so no need to return anything
                self.common_param_set(model, peak,error, y_max)
                # avoid using default parameter guess
                default_params = {
                    prefix+'center': peak[0],
                    prefix+'height': peak[1],
                    prefix+'sigma': x_range / len(x) * np.min(peak_widths)#peak[2]
                }
            else:
                raise NotImplemented(f'model {basis_func["type"]} not implemented yet')
            
            if 'help' in basis_func:  # allow override of settings in parameter
                for param, options in basis_func['help'].items():
                    model.set_param_hint(param, **options)
            
            # make model parameters based on the 
            model_params = model.make_params(**default_params, **basis_func.get('params', {}))
            # Add up each individual models and update their parameters accordingly
            composite_model, params = self.model_params_update(composite_model, model, params, model_params)                       

        return composite_model, params

    def fit(self, PLevol_obj, nframe=5, startIndex=2, verbose=True):
        """
        Args:
            PLevol_obj: a PLevol object, see class PLevol()
            nframe: number of frames of perform fitting over
            startIndex: the number of frame to start performing peak fitting
            verbose: whether or not to print out chi-sqr and plot fitting results
        
        Returns:
            None
        """
        self.output_series=[]
        self.first_pass=True
        for i in np.arange(0,nframe)+startIndex:
    
            index=i
            x = np.array(PLevol_obj.PLs[index].W)
            y = np.array(PLevol_obj.PLs[index].I - PLevol_obj.PLs[0].I)
            fig, ax = plt.subplots(1, 3, sharey=True, figsize=(15, 4))
            fig.suptitle('Frame '+str(i), fontsize=16)
            ax[0].scatter(x, y, s=4, c='b')
            
            output = None
            out1 = None
            out2 = None
            if self.first_pass:
                peaks_found, composite_model, params = self.peaks2spec(x, y)
                output = composite_model.fit(y, params, x=x)
            else:
                peaks_found, pk_model, pk_params, lt_model, lt_params = self.both2spec(x, y)
                out1 = pk_model.fit(y, pk_params, x=x)
                out2 = lt_model.fit(y, lt_params, x=x)
                if out1.chisqr <= out2.chisqr:
                    output=out1
                else:
                    output=out2
                if verbose: print("Frame", index ,"CWT Chi-sqr",out1.chisqr,"\t Last  Chi-sqr", out2.chisqr)
            
            if verbose:
                if not self.first_pass:
                    for peak in peaks_found:
                        ax[0].axvline(x=peak, c='black', linestyle='dotted')
                        
                    comps = out1.eval_components(x=x)
                    for each in comps:
                        ax[1].plot(x, comps[each], '--', label=each)
                    ax[1].legend()
                    ax[1].set_title("Peak Fitting with CWT Peak Parameters")

                    comps = out2.eval_components(x=x)
                    for each in comps:
                        ax[2].plot(x, comps[each], '--', label=each)
                    ax[2].legend()
                    ax[2].set_title("Peak Fitting with Last Fitting Parameters")
                else:
                    for peak in peaks_found:
                        ax[0].axvline(x=x[peak], c='black', linestyle='dotted')

                comps = output.eval_components(x=x)
                for each in comps:
                    ax[0].plot(x, comps[each], '--', label=each)
                ax[0].legend()
                ax[0].set_title("Best Fit")
                ax[1].scatter(x, y, s=4, c='y')
                ax[2].scatter(x, y, s=4, c='y')
                # plt.close()

            self.output_series.append(output)
            self.first_pass=False # other than the first iteration, use both2spec to find the best fit
    
    def peak_avg(self, out):
        """ Calculated peak height weighted average

        Args:
            out: a lmfit output object
        
        Returns:
            peak height weighted average
        """
        numer=0
        denom=0
        for each in out.params:
            if each.find("center") != -1:
                denom += out.params[each[:3]+'height'].value
                numer += out.params[each].value*out.params[each[:3]+'height'].value
        return numer/denom

    def plot_peakAvg(self):
        """ Plot a series of peak height weighted average from self.output_series

        Args:
            None
        
        Returns:
            None
        """
        fig, ax = plt.subplots(1,1)
        self.avgSeries = [self.peak_avg(each) for each in self.output_series]
        ax.scatter(np.arange(len(self.avgSeries)), self.avgSeries, label="Peak-Height-Weighted")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_ylabel("Frame Number")
        ax.set_title("Peak-Height-Weighted Wavelength Average")