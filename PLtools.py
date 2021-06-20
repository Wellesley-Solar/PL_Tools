# Import required packages
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import curve_fit
# from PLfunctions import sample_name, find_nearest, weighted_PL, trim_data, exp_fit
import scipy.optimize 
from lmfit import models
from numpy import random
from scipy import signal
from BaselineRemoval import BaselineRemoval
import seaborn as sn


c_light = 2.99792458e8 * 1e9 #
h_plank = 4.1357e-15 

def Wavelen2eV(wavelen):
    """
    wavelen: array like, an array of e&m wavelength in nm
    """
    eV_tmp=[c_light/each*h_plank for each in wavelen]

    return np.array(eV_tmp)


class PLspec:
    """
    PLspec accepts the path to a .csv file containing spectrometer output with two columns 
    named "Wavelength" and "Intensity." The wavelength series is saved in PLspec.W, the 
    intensity series in PLspec.I, the photon energy series corresponding to the wavelength 
    in PLspec.I. The intensity-weighted wavelength/photon energy over time is also calculated
    upon initializing and saved in PLspec.IweightedW or PLspec.IweightedE.
    
    """
    def __init__(self, file, xy_name=["Wavelength", "Intensity"], mod="Z"):
        """Initiates the PLspec class by reading a spectrum csv into a dataframe
        
        Args:
            file: path to a .csv file containing spectrometer output with at least two columns that
            are named "Wavelength" and "Intensity." Intensity weight averages for wavelength and 
            photon energy are made available

        Returns:
            None
        
        """
        self.file = file
        self.og = pd.read_csv(self.file, index_col=None, header=0).rename(columns={xy_name[0]: "W", xy_name[1]: "I"})
        baseObj=BaselineRemoval(self.og["I"])
        if mod=="Z":
            self.og["I"]=baseObj.ZhangFit(lambda_=1e4,porder=3,itermax=30)
        elif mod=="I":
            self.og["I"]=baseObj.IModPoly(15)
        
        self.df = self.og
        self.format_data()
        

    def format_data(self):
        """Perform basic intensity weighted average calculations and save data to appropriate attributes
        
        Args:
            None
        
        Returns:
            None
        
        """
        # self.df = self.df.rename(columns={"Wavelength": "W", "Intensity": "I"})
        # self.df["W"] = self.df["Wavelength"]
        # self.df["I"] = self.df["Intensity"]
        self.df["E"] = c_light/self.df["W"]*h_plank
        # print(self.df["I"], type(self.df["I"].values))
        self.sum_cnt = np.log(sum(self.df["I"]))
        self.IweightedW = sum(self.df["I"]*self.df["W"])/sum(self.df["I"])
        self.IweightedE = sum(self.df["I"]*self.df["E"])/sum(self.df["I"])
        # Weighted Variance
        self.IstdW = np.sqrt(sum(self.df["I"]*(self.df["W"] - self.IweightedW)**2)/sum(self.df["I"]))
        self.IstdE = np.sqrt(sum(self.df["I"]*(self.df["E"] - self.IweightedE)**2)/sum(self.df["I"]))
        self.IskewW = sum(self.df["I"]*(self.df["W"] - self.IweightedW)**3)/sum(self.df["I"])
        self.IskewE = sum(self.df["I"]*(self.df["E"] - self.IweightedE)**3)/sum(self.df["I"])
        self.df = self.df.reset_index(drop=True)
        
    def plot(self, mode="E"):
        """Plot the spectrum in the format of intensity versus wavelength
        
        Args:
            mode: str, the x-axis of the plot is always time, choosing the y-axis to
            be either "E" for photon energy, or "W" for nanometer wavelength
        
        Returns:
            None
        
        """
        fig, ax = plt.subplots()
        ax.scatter(self.df[mode], self.df["I"], s=4)
        ax.set_xlabel(mode)
        ax.set_ylabel("Intensity(Counts)")
        ax.set_title(self.file, {'fontsize': "small",})
        
        
    def retore(self):
        """Remove the cutoff for the spectrum data and restore the data back to its full length
        
        Args:
            None
        
        Returns:
            None
        
        """
        self.df = self.og 
        self.format_data()

        
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
        self.minCond = self.df["W"]>self.Wmin
        self.maxCond = self.df["W"]<self.Wmax
        self.df = self.df[self.minCond & self.maxCond]
        self.format_data()
        
    
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

    # def sub_bline(self, mod="Z"):
    #     baseObj=BaselineRemoval(self.I)
    #     if mod=="Z":
    #         self.I=baseObj.ZhangFit(lambda_=1e4,porder=3,itermax=30)
    #     elif mod=="I":
    #         self.I=baseObj.IModPoly(3)
    #     self.IweightedW = sum(self.I*self.W)/sum(self.I)
    #     self.IweightedE = sum(self.I*self.E)/sum(self.I)

class PLevol:
    def __init__(self, folder, t, xy_name=["Wavelength", "Intensity"], bl_mod="Z"):
        """Initiates the PLevol class by converting full spectrum csv into individual PLspec class.
        A series of intensity weighted wavelength over time and another series of intensity weighted
        photon energy over time is calculated.
        
        Args:
            folder: the path to folder/directory in which spectrometer files are stored
        
        Returns:
            None
        
        """
        self.og = [PLspec(file, xy_name=xy_name, mod=bl_mod) for file in sorted(glob.glob(folder+"/*.csv"))]
        self.PLs = self.og
        self.t = t
        self.df = self.format_data()
        self.S = np.matrix([each.df["I"].values for each in self.PLs])

    def format_data(self):
        self.df_alt = pd.DataFrame(
            np.transpose([PL.df.I.values for PL in self.PLs]), columns=np.arange(len(self.PLs)).astype(str)
        )
        self.df_alt["E"] = self.PLs[0].df.E
        return pd.DataFrame(
            {
                "t": np.array(self.t),
                "W": np.array([each.IweightedW for each in self.PLs]),
                "E": np.array([each.IweightedE for each in self.PLs]),
                "sum_cnt": np.array([each.sum_cnt for each in self.PLs]),
                "IstdW": np.array([each.IstdW for each in self.PLs]),
                "IstdE": np.array([each.IstdE for each in self.PLs]),
                "IskewW": np.array([each.IskewW for each in self.PLs]),
                "IskewE": np.array([each.IskewE for each in self.PLs]),
            })


    def ridge(self, n=5, fset=0.15, alpha=0.3, startIndex=0):
        """ Making ridgeline plots across a set of PL frame
        Args:
            n: int, the number of frames to display in a ridgeline plot
            fset: float, vertical offset between individuals frames in ridgeline plot
            alpha: float, the transparency of the ridgeline fill-ins
            and the first element specifies the name of the y-data column
            startIndex: int, which frame to start making ridgelines
        Returns:
            None
        """
        fig, ax = plt.subplots()
        i_frames = startIndex+np.linspace(0,len(self.df_alt.columns)-1, n, endpoint=False,)
        i_frames = i_frames[::-1]
        for i, v in enumerate(i_frames):
            # ii = len(self.df_alt.columns)-2-int(v)
        #     print(v)
            ax.plot(self.df_alt.E, self.df_alt[str(int(v))]/max(self.df_alt[str(int(v))])-fset*i,"")
            ax.fill_between(
                self.df_alt.E, self.df_alt[str(int(v))]/max(self.df_alt[str(int(v))])-fset*i,
                -fset*i,
                alpha=alpha,
                label="Frame "+str(int(v)),
            )
        ax.axes.get_yaxis().set_visible(False)
        ax.set_xlabel("PL Emission (eV)")
        ax.set_title("Normalized Intensity (a.u.)")
        ax.legend()


    def BH_cosmic_remove(self, cosmic_prom=24, startIndex=0, max_pixel=10):
        """ Remove sharp peaks possibly due to cosmic background radiation in the data. For the explanation on
        the implementation, please refer to the section in the corresponding thesis
        Args:
            cosmic_prom: int, prominence threshold of cosmic ray. It multiplies the individual datapoints' differences
            between two most similar frames
            startIndex: int, which frame to start making ridgelines
            max_pixel: int, the maximum number of data points in a frame allowed to be deemed
            to be sharp enough for cosmic ray
        Returns:
            None
        """
        self.S = np.matrix([each.df["I"].values for each in self.PLs])
        self.mat_bar = np.matrix([
            signal.savgol_filter(each.df["I"].values, window_length=5, polyorder=1) for each in self.PLs
        ])
        self.covar_mat = self.S * self.S.T
        self.numer = np.multiply(self.covar_mat, self.covar_mat)
        self.C_nm = np.zeros([self.S.shape[0], self.S.shape[0]])
        for i in range(self.C_nm.shape[0]):
            for j in range(self.C_nm.shape[1]):
                if i == j:
                    self.C_nm[i, j] = 0
                    continue
                self.C_nm[i,j] = self.numer[i,j]/(self.covar_mat[i,i]*self.covar_mat[j,j])

        self.sigma_n = float(np.median(self.noise_std_est(self.S, self.mat_bar)))
        self.S_p = self.sim_spec_res(self.S, startIndex=startIndex)
        self.S_after = np.zeros(self.S.shape)
        self.S_swap = np.zeros(self.S.shape)
        self.mat_diff = self.S - self.S_p 
        for i, row in enumerate(self.mat_diff):
            if sum(np.array(row)[0] > cosmic_prom * float(self.sigma_n)) > max_pixel:
                self.S_after[i, :] = self.S[i, :]
                continue
            for j, col in enumerate(np.array(row)[0].tolist()):
                if col <= cosmic_prom * self.sigma_n:
                    self.S_after[i, j] = self.S[i, j]
                    # continue
                elif col > cosmic_prom * self.sigma_n:
                    self.S_after[i, j] = self.S_p[i, j]
                    self.S_swap[i, j] = 1#self.mat_diff[i, j]
        for i, row in enumerate(self.S_after):
            df_tmp = pd.DataFrame(
                {
                    "W": self.PLs[i].df["W"],
                    "I": np.array(row)
                }
            )
            self.PLs[i].df = df_tmp
            self.PLs[i].format_data()
        self.df = self.format_data()


    def heatmap(self, matrix, figsize = (20, 10), cmap="hot"):
        """ Plot the entire timeseries of PL intensities in the form of a heatmap

        Args:
            matrix: 2d-array like, whose rows are individual frames and whose columns correspond to PL wavelength
            figsize: int, figure size of heatmap
            cmap: str, colormap style for the heatmap

        Returns:
            None
        """
        fig, scatter = plt.subplots(figsize = (20,10))
        sn.heatmap(matrix, cmap="hot")

    def noise_std_est(self, m1, m2):
        """ Estimate the noise level of the measurement.

        Args:
            m1: 1d-array like, individual frame intensity of a row
            m2: 1d-array like, individual frame intensity of a different row
        
        Returns:
            None
        """
        mat_diff = m1 - m2
        sig_n = np.zeros((m1.shape[0], 1))
        for i,_ in  enumerate(m1):
            sig_n[i] = np.sqrt(
                mat_diff[i,:] * mat_diff[i,:].T
            )/m1.shape[0]# number of frames
        return sig_n


    def sim_spec_res(self, m1, startIndex=0):
        """ swap the rows of the input matrix with its most similar rows

        Args:
            m1: 2d-array like
        
        Returns:
            a matrix whose column has been reordered
        """
        sim_spec = np.arange(m1.shape[0])
        for frame in np.arange(0, m1.shape[0]-startIndex)+startIndex:
            tmp=np.where(self.C_nm[frame,:]==max(self.C_nm[frame,:]))[0]
            sim_spec[frame] = int(tmp)
        return m1[sim_spec, :]


    def restore(self):
        """Remove the cutoff for the spectrum data and restore the data back to its full length
        
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
        self.df["W"] = np.array([each.IweightedW for each in self.PLs])
        self.df["E"] = np.array([each.IweightedE for each in self.PLs])
        self.df["sum_cnt"] = np.array([each.sum_cnt for each in self.PLs])


    # def sub_baseline(self, mod="Z"):
    #     [PL.sub_bline(mod) for PL in self.og]
    #     self.df_Wavg = [each.IweightedW for each in self.PLs]
    #     self.df_Eavg = [each.IweightedE for each in self.PLs]

    def df_plot(self, mode="W"):
        """ Plot a series of peak height weighted average from self.output_series

        Args:
            mode: str, the x-axis of the plot is always time, choosing the y-axis to
            be either "E" for photon energy, or "W" for nanometer wavelength
        
        Returns:
            None
        """
        fig, ax = plt.subplots(1,1)
        ax.scatter(self.df["t"], self.df[mode], label="Spectrum-Weighted")
        ax.set_ylabel(mode)
        ax.set_xlabel("Time (s)")
       

# Some of the following peak fitting code was originally written by Chris Ostrouchov
# You can find them here: https://chrisostrouchov.com/post/peak_fit_xrd_python/
# Qingmu Deng rewrote most of the original code for PL Fitting for the Belis Lab at Wellesley College
class Fitter():
    def __init__(self, PLevol_obj, bd_l_w=600, bd_u_w=800, thres_amp=3,
                 bd_l_sigma=5, bd_u_sigma=40, min_amp=1e-6,
                 wiggle=10, num_peak=4, peak_widths=(10, 200), 
                 model='GaussianModel', ftol=1e-10
                ):
        self.output_series = []
        self.first_pass = True
        self.bd_l_w = bd_l_w
        self.bd_u_w = bd_u_w
        self.thres_amp = thres_amp
        self.bd_l_sigma = bd_l_sigma
        self.bd_u_sigma = bd_u_sigma
        self.min_amp = min_amp
        self.wiggle = wiggle
        self.num_peak = num_peak
        self.PLevol_obj = PLevol_obj
        self.peak_widths = peak_widths
        self.model = model
        self.ftol = ftol

    def basicparam(self, spec):
        x = spec['x']
        y = spec['y']
        x_min = np.min(x)
        x_max = np.max(x)
        x_range = x_max - x_min
        y_max = np.max(y)
        return None, None, x, y, x_min, x_max, x_range, y_max


    def spec_gen(self, x, y, peaks):
        tmp= [{'type': self.model} for i in range(self.num_peak)]
        return {
            'x': x,
            'y': y,
            'model': tmp#len(peaks))]
            }


    def common_param_set(self, mdl, peak, y_max):
        """ Specify bounds for relevant model parameters
        Args:
            mdl: a lmfit model object
            peak: a list of [peak_x_value, peak_y_value]
            wiggle: freedom a peak's center is given to the left and right in nm
            y_max: maximum y of the spectrum
        
        Returns:
            None
        """
        # mdl.set_param_hint('sigma', min=5)
        # if self.zero_bg:
        mdl.set_param_hint('sigma', min=self.bd_l_sigma, max=self.bd_u_sigma)
        mdl.set_param_hint('center', min=peak[0]-self.wiggle, max=peak[0]+self.wiggle)
        mdl.set_param_hint('height', min=self.min_amp, max=1.1*y_max)
        mdl.set_param_hint('amplitude', min=self.min_amp)
        

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


    def peaks2spec(self, x, y):
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
        peak_indicies = signal.find_peaks_cwt(y, self.peak_widths)
    #     print(peak_index)
        temp = []
        # print(x)
        print(peak_indicies)
        for peak_index in peak_indicies:
            if x[peak_index] > self.bd_l_w and x[peak_index] < self.bd_u_w:
                temp.append(peak_index)
        peak_indicies = np.array(temp)
        spec = self.spec_gen(x, y, peak_indicies)
        print(peak_indicies)
        
        composite_model, model_params = self.my_modelgen(spec, peak_indicies)

        return peak_indicies, composite_model, model_params
    

    def my_modelgen(self, spec, peak_indicies):
        """ Generate an initial composite model and model parameters based on CWT peak detection
        Args:
            spec: initial model specification variable, see spec_gen()
            peak_indicies: peak indices identified by CWT
            peak_widths: allowable peak width for CWT search
            wiggle: freedom a peak's center is given to the left and right in nm
        
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
                model.set_param_hint('sigma', min=self.bd_l_sigma, max=self.bd_u_sigma)
                model.set_param_hint('center', min=x[peak_index]-self.wiggle, max=x[peak_index]+self.wiggle)
                model.set_param_hint('height', min=self.min_amp, max=1.1*y_max)
                model.set_param_hint('amplitude', min=self.min_amp)
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
                                 
                                    
                                    
    def both2spec(self, x, y):
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
        peak_indicies = signal.find_peaks_cwt(y, self.peak_widths)
        peaks_cwt = []
        for peak_index in peak_indicies:
            if x[peak_index] > self.bd_l_w and x[peak_index] < self.bd_u_w:
                peaks_cwt.append([x[peak_index],y[peak_index]])
        
        # Consider the peak positions from the last fitting routine as well
        last_peaks = []
        if self.output_series != []:
            last_out=self.output_series[-1]
            for each in last_out.params:
                if each.find("center") != -1:
                    amp = last_out.params[each[:3]+'height']
                    # if amp.value < self.thres_amp:
                    #     continue
                    last_peaks.append([last_out.params[each].value, last_out.params[each[:3]+'height'].value,\
                                    last_out.params[each[:3]+'sigma'].value])
        
        # Generate a first model based on CWT-extracted peaks
        pk_spec = self.spec_gen(x, y, peaks_cwt)
        pk_model, pk_params = self.cwt_modelgen(pk_spec, peaks_cwt)
        # Generate a second model based on last peakfitting parameters
        lt_spec = self.spec_gen(x, y, last_peaks)
        lt_model, lt_params = self.last_modelgen(pk_spec, last_peaks)
        
        # Get the CWT peak locations for comparison
        peaks_cwt = [each[0] for each in peaks_cwt]
        return peaks_cwt, pk_model, pk_params, lt_model, lt_params


    def last_modelgen(self, spec, peaks):
        """ Generate a composite model and model parameters based on last fitting parameters
        Args:
            spec: initial model specification variable, see spec_gen()
            peak_indicies: peak indices identified by CWT
            peak_widths: allowable peak width for CWT search
            wiggle: freedom a peak's center is given to the left and right in nm
        
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
                self.common_param_set(model, peak, y_max)

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
                                 

    def cwt_modelgen(self, spec, peaks):
        """ Generate a composite model and model parameters based on CWT peak detections
        Args:
            spec: initial model specification variable, see spec_gen()
            peak_indicies: peak indices identified by CWT
            peak_widths: allowable peak width for CWT search
            wiggle: freedom a peak's center is given to the left and right in nm
        
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
                self.common_param_set(model, peak, y_max)
                # avoid using default parameter guess
                default_params = {
                    prefix+'center': peak[0],
                    prefix+'height': peak[1],
                    prefix+'sigma': x_range / len(x) * np.min(self.peak_widths)#peak[2]
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


    def fit(self, nframe=5, startIndex=2, zero_bg=True, verbose=True, mthd="lbfgsb", neval=10):
        """
        Args:
            PLevol_obj: a PLevol object, see class PLevol()
            nframe: number of frames of perform fitting over
            startIndex: the number of frame to start performing peak fitting
            verbose: whether or not to print out chi-sqr and plot fitting results
        
        Returns:
            None
        """
        self.output_series = []
        self.param_num = []
        self.zero_bg = zero_bg
        self.first_pass = True
        for i in np.arange(0,nframe-startIndex)+startIndex:
    
            index = i
            x = np.array(self.PLevol_obj.PLs[index].df["W"])
            if self.zero_bg:
                y = np.array(self.PLevol_obj.PLs[index].df["I"] - self.PLevol_obj.PLs[0].df["I"])
            else:
                y = np.array(self.PLevol_obj.PLs[index].df["I"])# - 635
            
            output = None
            out1 = None
            out2 = None
            if self.first_pass:
                peaks_found, composite_model, params = self.peaks2spec(x, y)
                output = composite_model.fit(y, params, x=x, method=mthd, max_nfex=neval)
            else:
                peaks_found, pk_model, pk_params, lt_model, lt_params=self.both2spec(x, y)
                out1 = pk_model.fit(y, pk_params, x=x, method=mthd, max_nfex=neval, ftol=self.ftol)
                out2 = lt_model.fit(y, lt_params, x=x, method=mthd, max_nfex=neval, ftol=self.ftol)
                # out12diff=math.abs(out2.chisqr-out1.chisqr)
                # if out12diff/min(out2.chisqr,out1.chisqr) < 0.1:

                if out1.chisqr <= out2.chisqr:
                    output=out1
                else:
                    output=out2
                if verbose: print("Frame", index ,"CWT Chi-sqr",out1.chisqr,"\t Last  Chi-sqr", out2.chisqr)
                
            # print(peaks_found)
            if verbose:
                fig, ax = plt.subplots(2, 2, sharey=False, figsize=(15, 12))
                fig.suptitle('Frame '+str(i), fontsize=16)
                ax[0,0].scatter(x, y, s=4, c='b')
                if not self.first_pass:
                    ax[1,0].scatter(x, y, s=4, c='y')
                    ax[1,1].scatter(x, y, s=4, c='y')
                    for peak in peaks_found:
                        ax[0,0].axvline(x=peak, c='black', linestyle='dotted')
                        
                    comps = out1.eval_components(x=x)
                    for each in comps:
                        ax[1,0].plot(x, comps[each], '--', label=each)
                    ax[1,0].legend()
                    ax[1,0].set_title("Peak Fitting with CWT Peak Parameters")

                    comps = out2.eval_components(x=x)
                    for each in comps:
                        ax[1,1].plot(x, comps[each], '--', label=each)
                    ax[1,1].legend()
                    ax[1,1].set_title("Peak Fitting with Last Fitting Parameters")
                else:
                    for peak in peaks_found:
                        ax[0,0].axvline(x=x[peak], c='black', linestyle='dotted')

                comps = output.eval_components(x=x)
                first_m=True
                for each in comps:
                    ax[0,0].plot(x, comps[each], '--', label=each)
                    if first_m:
                        accu_comps = comps[each]
                        first_m = False
                    else:
                        accu_comps += comps[each]
                ax[0,0].plot(x, accu_comps, 'r--', label="sum")
                ax[0,0].legend()
                ax[0,0].set_title("Best Fit")
                ax[0,1].plot(x, 0*np.arange(len(x)), 'k-', label="sum", alpha=0.3)
                ax[0,1].plot(x, y-accu_comps, 'ro', label="sum")
                ax[0,1].plot(x, y-accu_comps, 'b-', label="sum", alpha=0.1)
                # ax[0,1].legend()
                ax[0,1].set_title("Residual " + str(sum(np.multiply(y-accu_comps,y-accu_comps))))
                # plt.close()

            ## Print out 
            if verbose:
                for name, param in output.params.items():
                    # print(name, param)
                    if "center" in name:
                        if param.stderr == None:
                            print('{:7s}    mean:{:8.5f}'.format(name, param.value))
                            continue
                        print('{:7s}    mean:{:8.5f}    std: {:8.5f}'.format(name, param.value, param.stderr))
            
            self.output_series.append(output)
            self.first_pass = False # other than the first iteration, use both2spec to find the best fit
            self.param_num.append(len(output.params))
        self.post_process()
        avgSeriesW = np.array([self.peak_avg(each) for each in self.output_series])
        avgSeriesE = Wavelen2eV(avgSeriesW)
        self.df=pd.DataFrame({
            "t": np.array(self.PLevol_obj.df["t"])[startIndex:nframe],
            "W": avgSeriesW,
            "E": avgSeriesE,
            # "min_w": self.min_w,
            # "max_w": self.max_w,
            # "min_e": Wavelen2eV(self.min_w),
            # "max_e": Wavelen2eV(self.max_w),
            "param_num": self.param_num,
            "sum_cnt": np.array(self.PLevol_obj.df["sum_cnt"])[startIndex:nframe],
            "IstdW": np.array(self.PLevol_obj.df["IstdW"])[startIndex:nframe],
            "IstdE": np.array(self.PLevol_obj.df["IstdE"])[startIndex:nframe]
        })
        # print(len(np.array(self.PLevol_obj.df["sum_cnt"])[startIndex:nframe]),len(avgSeriesW))

    def post_process(self):
        self.min_w = []
        # for output in self.output_series:
        #     params = output.params
        #     tmp_ctr = []
        #     for i,v in enumerate(params):
        #         if "center" in v:
        # #             print(i, v, params[v].value,params[v].stderr)
        #             tmp_ctr.append([ v, params[v].value,params[v].stderr])
        #     tmp_ctr.sort(key = lambda x: x[1],reverse=False)
        #     self.min_w.append(tmp_ctr[0][1])
        self.mid_w = []
        self.max_w = []
        self.peaks = []
        # self.peaks_h = []
        for output in self.output_series:
            # params = output.params
            tmp_ctr = []
            tmp_ht = []
            for name, param in output.params.items():
            # for i,v in enumerate(params):
                if "center" in name:
                    if output.params[name[:3]+'height'].value < self.thres_amp:
                        continue
        #             print(i, v, params[v].value,params[v].stderr)
                    tmp_ctr.append([param.value, output.params[name[:3]+'height'].value])
                    # tmp_ht.append()
            # tmp_ctr.sort()
            # self.min_w.append(tmp_ctr[0])
            # self.max_w.append(tmp_ctr[-1])
            # if len(tmp_ctr) < 3:
            #     self.mid_w.append(tmp_ctr[0])
            # if len(tmp_ctr) == 3:
            #     self.mid_w.append(tmp_ctr[1])
            tmp_ctr = np.array(tmp_ctr)
            self.peaks.append(tmp_ctr)

    
    def peak_avg(self, out):
        """ Calculated peak height weighted average

        Args:
            out: a lmfit output object
        
        Returns:
            peak area weighted average
        """
        numer = 0
        denom = 0
        for each in out.params:
            if each.find("center") != -1:
                # Calculate area weighted average from fit results
                denom += out.params[each[:3]+'height'].value*out.params[each[:3]+'sigma'].value
                numer += out.params[each].value*out.params[each[:3]+'height'].value*out.params[each[:3]+'sigma'].value
        return numer/denom

    def plot(self, mode="E"):
        """ Plot a series of peak height weighted average from self.output_series

        Args:
            mode: str, the x-axis of the plot is always time, choosing the y-axis to
            be either "E" for photon energy, or "W" for nanometer wavelength
        
        Returns:
            None
        """
        
        fig, ax = plt.subplots(1,1)
        # if mode=="W":
        ax.scatter(self.df["t"], self.df[mode], label="Peak-Height-Weighted")
        # if mode=="E":
        #     ax.scatter(np.arange(len(self.avgSeriesE)), self.avgSeriesE, label="Peak-Height-Weighted")
        # elif mode=="min":
        #     ax.scatter(np.arange(len(self.avgSeries)), self.min_w, label="Peak")
        # elif mode=="max":
        #     ax.scatter(np.arange(len(self.avgSeries)), self.max_w, label="Peak")
        # ax.set_ylabel()
        ax.set_xlabel("Frame Number")
        ax.set_title(mode)