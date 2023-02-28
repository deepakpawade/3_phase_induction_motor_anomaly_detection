import numpy as np
import copy
import os
import pandas as pd


class GetFeatures:

    def __init__(self,df,fundamental_frequency = 50, sampling_frequency = 10000):
        self.df = copy.deepcopy(df)
        self.fs = sampling_frequency
        self.funda_freq = fundamental_frequency
        self.fft_data = self.getfft(self.df)
        feature_vector = [self.get_features(self.df)] 

    def _getfft(self,df):
        return np.fft.fft(df, axis=0)

    def _get_features(self,df):

        # fft_data = np.fft.fft(df, axis=0)
        fft_data = self._getfft(df)
        # Set sampling frequency
        


        # Compute power spectral density
        # This line computes the power spectral density (PSD) of the FFT by taking the absolute value of the FFT, squaring it, and dividing by the length of the current_data array. The PSD represents the distribution of power over the frequency range.
        psd = np.abs(fft_data) ** 2 / len(df)


        # Set frequency vector
        # This line computes the frequency vector for the FFT by using the fftfreq() function from NumPy, which generates a vector of frequencies that correspond to each element of the FFT output.
        freqs = np.fft.fftfreq(len(fft_data), 1 / self.fs) 

        # Identify the fundamental frequency and its harmonics
    
        harmonic_magnitudes = self.get_harmonic_magnitudes(self.fft_data,self.fs, self.funda_freq)
  
        # Calculate THD
        fundamental_index = int(self.fundamental_freq * (len(fft_data)/2) / self.fs)
        thd = np.sqrt(np.sum(np.abs(fft_data[fundamental_index-2:fundamental_index+3])**2) - np.abs(fft_data[fundamental_index])**2) / np.abs(fft_data[fundamental_index])

        # Calculate RMS current
        rms_current = np.sqrt(np.mean(df**2))

        # Calculate current unbalance
        phase_angles = np.angle(fft_data[1:4, :], deg=True)
        phase_angles_diff = np.diff(phase_angles, axis=0)
        current_unbalance = np.max(np.abs(phase_angles_diff))

        return fft_data,harmonic_magnitudes,thd,rms_current,current_unbalance

    def get_harmonic_magnitudes(fft_data, fs = 10000, ffreq=50 ):
        fs = 10000
        fundamental_freq = ffreq # or 60 Hz depending on the AC power supply frequency
        harmonic_freqs = [2*fundamental_freq, 3*fundamental_freq, 4*fundamental_freq] # extract 2nd, 3rd and 4th harmonics
        harmonic_indices = [int(freq/fs * (len(fft_data)/2)) for freq in harmonic_freqs] # compute indices of harmonic components
        # harmonic_indices = [int(freq/fundamental_freq * (len(fft_1)//2)) for freq in harmonic_freqs if freq < fs/2] 

        harmonic_magnitudes = np.abs(fft_data[harmonic_indices]) # extract magnitudes of harmonic components
        return harmonic_magnitudes