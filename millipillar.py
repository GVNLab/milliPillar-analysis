# Last updated in 11/06/2020 by Youngbin Kim
import scipy as sp
from scipy import signal
import pandas as pd
import numpy as np
from skimage import io
import glob
import os.path

class Video():
    def __init__(self, file_path, filetype = 'tif', frame_rate = None):
        # for nd2 need to give full .nd2 filepath for nd2
        # for tiff, give the folder name and program will combine the tiff images
        if filetype == 'nd2':
            from nd2reader import ND2Reader
            self.raw_video = ND2Reader(file_path)
            self.frame_rate = self.raw_video.frame_rate
        elif filetype == 'tif':
            self.raw_video = io.imread(file_path)
            if frame_rate: # if frame rate 
                self.frame_rate = frame_rate
            else:
                # only using this pycromanager class for metadata
                # because raw data handling is super slow
                # reason is that scipy loads the whole thing to RAM (good for computational speed)
                # while pycromanager loads into dask format, which loads each frame as needed (good for low RAM)
                from pycromanager import Dataset
                dataset = Dataset(file_path)
                self.frame_rate = float(dataset.read_metadata(time=0)['Andor sCMOS Camera-FrameRate'])

        self.trace = np.mean(self.raw_video, axis=(1,2))


class Trace():
    def __init__(self, data, sampling_rate, name = None):
        self.data = data# raw data
        self.name = name
        self.peaks = None # this should be a pandas dataframe
        self.valleys = None # contains indices of points used for baseline fitting
        self.bpm = None # peak detector or fourier??
        self.sampling_rate = sampling_rate
        # i think peak detector is better for irregular beating
        self.avg = None # avg peak width, height, F/F0, etc.
        # group by each peak
        self.tau = None
        self.fwhm = None
        self.fw90m = None
        # group everythign into a dataframe indiv peak info vs whole
        self.baseline = None
        self.normalized = None
        self.rr_interval = None # length is len(self.peaks) - 1
        # heart rate variability metrics
        # see Table 1 of this paper to understand what they mean
        # https://www.frontiersin.org/articles/10.3389/fpubh.2017.00258/full
        self.SDRR = None
        self.RMSSD = None
        self.pRR50 = None # as opposed to pNN50, pRR50 includes abnormal beats as well
        self.pRR20 = None
        self.max_data = None
        self.max_df_f0 = None

    def normalize(self, array):
            result = array.copy()
            max_value = array.max()
            min_value = array.min()
            result = (array - min_value) / (max_value - min_value)
            return result
    def analyze(self, max_bpm=360, min_prominence = .25, baseline_fit="exp", baseline_fit_range = None):
        
        # hyperparameters
        # max_bpm: sets min number of frames between peaks for peak_finder
        # min_prominence: prominence (height from baseline) of peak must be greater than min_prominence
        
        # calculate_baseline finds peaks and denoises the signal)
        self.calculate_baseline(max_bpm, min_prominence=0.2*(max(self.data) - min(self.data)), method=baseline_fit, baseline_fit_range = baseline_fit_range)

        # recalculate peaks after removing baseline
        distance = 1 /(max_bpm / 60) * self.sampling_rate # for example, 250bpm is 4.17 Hz which is 240ms between beats. this * fps =  min # frames between beats
        self.peaks, properties = sp.signal.find_peaks(self.normalized, distance = max(1,distance), prominence=min_prominence, width=0, rel_height =1)
        
        # calculate contraction and relaxation velocity
        width50 = sp.signal.peak_widths(self.normalized, self.peaks, rel_height=0.5)
        self.fwhm = width50[0] / self.sampling_rate # full width half max is default. Contains 3 elements: FWHM, left WHM, right WHM
        self.contract50 = (self.peaks - width50[2]) / self.sampling_rate
        self.relax50 = (width50[3] - self.peaks) / self.sampling_rate
        
        width90 = sp.signal.peak_widths(self.normalized, self.peaks, rel_height=0.9)
        self.fw90m = width90[0] / self.sampling_rate # full/left/right width at 90% of peak. Contains 3 elements: FW90M, left W90M, right W90M
        self.contract90 = (self.peaks - width90[2]) / self.sampling_rate
        self.relax90 = (width90[3] - self.peaks) / self.sampling_rate

        # calculate tau
        decay_frame = sp.signal.peak_widths(self.normalized, self.peaks, rel_height=1-np.exp(-1))[3]
        self.tau =  (decay_frame - self.peaks) / self.sampling_rate
        
        # calculate RMSSD and pRR50
        self.rr_interval = np.diff(self.peaks) / self.sampling_rate
        self.SDRR = self.rr_interval.std()
        self.RMSSD = np.sqrt(np.mean(self.rr_interval ** 2))
        if self.sampling_rate >= 40: # we need at least 40 fps by Nyquist sampling theorem (2 * 1/50ms)
            diff_rr = np.abs(np.diff(self.rr_interval))
            self.pRR50 = (diff_rr > 0.05).sum() / len(diff_rr)
        
        # calculate bpm 
        if len(self.peaks) > 1: # given there are 2 or more peaks
            self.bpm = (len(self.peaks) - 1) / (self.peaks[-1] - self.peaks[0]) \
                * self.sampling_rate * 60 # convert beats per frame to beats per minute
        # calculate max df/f0
        self.max_data = np.max(self.data)
        try:
            self.max_df_f0 = np.max(self.df_f0)
        except:
            self.max_df_f0 = None
        
    def calculate_baseline(self, max_bpm, min_prominence, method="exp", baseline_fit_range = None):        
        if method is None:
            self.baseline = np.zeros(len(self.data))
            self.normalized = self.normalize(self.data)

        else:
            if baseline_fit_range:
                fit_data = self.data[baseline_fit_range[0]: baseline_fit_range[1]]
            else:
                fit_data = self.data
            # updates self.baseline, self.valleys, and self.normalized
            distance = 1 /(max_bpm / 60) * self.sampling_rate # for example, 250bpm is 4.17 Hz which is 240ms between beats. this * fps =  min # frames between beats

            # find valleys to use for our baseline fit
            self.valleys, _ = sp.signal.find_peaks(-fit_data, distance = max(1,distance), prominence=min_prominence, width=0, rel_height =1)
            # if less than 3 points, can't fit an equation with 3 unknowns
            # so we need to include more points near the baseline
            # this includes every point that's not within 95% of the peaks 
            if len(self.valleys) < 4:
                peaks, properties = sp.signal.find_peaks(fit_data, distance = distance, prominence=min_prominence, width=0, rel_height =0.95)
                valleys = np.full(len(fit_data), True)
                for i in range(len(peaks)):
                    valleys[int(properties['left_ips'][i]+1):int(properties['right_ips'][i])] = False # remove xrange within 95% height of peak
                self.valleys = np.where(valleys)[0]
            
            if method == "power":
                # this function from this paper fits super well
                # Interpretation of Fluorescence Decays using a Power-like Model
                # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1303114/
                def baseline_func(t, tau, q, cons):
                    return (2-q)/tau*(1-(1-q)*t/tau)**(1/(1-q))+cons
                try:
                    baseline_param, _ = sp.optimize.curve_fit(baseline_func, self.valleys, self.data[self.valleys], p0=[-100,-.5,.5],maxfev=100000)
                    tau, q, cons = baseline_param
                    self.baseline = baseline_func(np.arange(len(self.data)), tau, q, cons)
                    
                except RuntimeError:
                    method = 'linear'
                    
            if method == "exp":
                def exp_func(t, A, tau, beta, cons):
                    return A*np.exp(-(t/tau)**(1-beta))+cons
                #p0 is very important. else it will not converge
                cons0 = self.data[-1] # last point's y value
                A0 = self.data[0] - cons0 # 1st point's y value - last point's y value
                tau0 = 3500 # empirical 
                beta0 = 0.07 # empirical 
                try:
                    baseline_param, _ = sp.optimize.curve_fit(exp_func, self.valleys, self.data[self.valleys], p0=[A0, tau0, beta0, cons0],maxfev=10000000)
                    A, tau, beta, cons = baseline_param
                    self.baseline = exp_func(np.arange(len(self.data)), A, tau, beta, cons)
                except RuntimeError:
                    method = "linear"
                    
            if method =="aggr_linear":
                self.baseline = np.interp(np.arange(len(self.data)), self.valleys, self.data[self.valleys])
            
            if method == 'linear':
                slope, intercept = np.polyfit(self.valleys, self.data[self.valleys], 1)
                self.baseline = slope * np.arange(len(self.data)) + intercept

            self.normalized = self.normalize(self.data - self.baseline)
            self.df_f0 = (self.data-self.baseline)/self.baseline 
        

    def summary(self):
        info = pd.DataFrame()
        info['num peaks'] = [len(self.peaks)]
        info['bpm'] = [self.bpm]
        info['RMSSD'] = [self.RMSSD]
        info['pRR50'] = [self.pRR50]
        info['max data'] = [self.max_data]
        info['max df_f0'] = [self.max_df_f0]
        info['avg RR interval'] = [self.rr_interval.mean()]
        info['avg tau'] = [self.tau.mean()]
        info['avg fwhm'] = [self.fwhm.mean()]
        info['avg contract50'] = [self.contract50.mean()]
        info['avg relax50'] = [self.relax50.mean()]
        info['avg fw90m'] = [self.fw90m.mean()]
        info['avg contract90'] = [self.contract90.mean()]
        info['avg relax90'] = [self.relax90.mean()]
        info['SDRR'] = [self.rr_interval.std()]
        info['std tau'] = [self.tau.std()]
        info['std fwhm'] = [self.fwhm.std()]
        info['std contract50'] = [self.contract50.std()]
        info['std relax50'] = [self.relax50.std()]
        info['std fw90m'] = [self.fw90m.std()]
        info['std contract90'] = [self.contract90.std()]
        info['std relax90'] = [self.relax90.std()]
        if self.name is not None:
            info.index = [self.name]
        return info
    
    def peak_summary(self):
        info = pd.DataFrame()
        info['peak'] = self.peaks / self.sampling_rate
        info['peak_index'] = self.peaks
        info['tau'] = self.tau
        info['fwhm'] = self.fwhm
        info['contract50'] = self.contract50
        info['relax50'] = self.relax50
        info['fw90m'] = self.fw90m
        info['contract90'] = self.contract90
        info['relax90'] = self.relax90
        return info

def stimulation_analysis(filepath, regimen, filetype, frame_shift = 20, baseline_fit="exp"):
    video = Video(filepath, filetype=filetype)

    traces = []
    trace = Trace(video.trace, sampling_rate = video.frame_rate)
    trace.analyze(baseline_fit=baseline_fit, baseline_fit_range=[0,2200])
    for i, row in regimen.iterrows():
        if (i == 0) and (frame_shift < 0):
            trace_segment_data = trace.df_f0[0: int(row['Frame Stop'] + frame_shift)]
        elif i == len(regimen.index) and (row['Frame Stop'] + frame_shift > len(video.trace.data)):
            trace_segment_data = trace.df_f0[int(row['Frame Start']):]
        else:
            trace_segment_data = trace.df_f0[int(row['Frame Start'] + frame_shift): int(row['Frame Stop']) + frame_shift]
        trace_segment = Trace(trace_segment_data, sampling_rate = video.frame_rate)
        trace_segment.analyze(baseline_fit=None)
        traces += [trace_segment]
    table = regimen.loc[:,['Voltage (V)', 'Freq (Hz)', 'Period(s)']]
    table.loc[:,'num peaks'] = list(map(lambda t: len(t.peaks), traces)) 
    table.loc[:,'Tissue Freq (Hz)'] = list(map(lambda t: t.bpm/60 if t.bpm is not None else None, traces))
    table.loc[:,'Capture Ratio (stim freq / beat freq)'] = table.loc[:,'Freq (Hz)'] / table.loc[:,'Tissue Freq (Hz)']
    
    ##### edited this part
    table.loc[:, 'Flag'] = np.abs(1 - table.loc[:,'Capture Ratio (stim freq / beat freq)']) > 0.1 # flag frequency mismatch
    table.loc[1:, 'Flag'] = table.loc[1:, 'Flag'] | table.isna().any(axis=1).loc[1:] # flag NaN values except for unstimulated trace (row 0)
    et_index = table[~table['Flag']].loc[1:10].last_valid_index() # find the last index that's not flagged
    mcr_index = table[~table['Flag']].loc[11:].last_valid_index() # loc is weird in that both start and end index is inclusive
    
    if et_index is not None:
        et = table.loc[et_index, 'Voltage (V)']
    else:
        et = None
        
    if mcr_index is not None:
        mcr = table.loc[mcr_index, 'Freq (Hz)']
    else:
        mcr = None
        
    table.loc[:,'trace'] = traces
    
    return trace, traces, table, et, mcr