import numpy as np
import copy
import os
import pandas as pd
from typing import Dict
from sklearn.tree import DecisionTreeClassifier

class ElectricalFeatureExtractor:
    """
    A class for calculating features from current waveform data.

    Parameters:
        df (pandas.DataFrame): The current waveform data as a pandas DataFrame.
        fundamental_frequency (float): The fundamental frequency of the current waveform.
        sampling_frequency (float): The sampling frequency of the current waveform data.

    Attributes:
        df (pandas.DataFrame): The current waveform data as a pandas DataFrame.
        fs (float): The sampling frequency of the current waveform data.
        funda_freq (float): The fundamental frequency of the current waveform.
        fft_data (numpy.ndarray): The FFT of the current waveform data.
        feature_vector (list): A list containing the feature vector calculated from the current waveform data.

    Methods:
        _getfft: Calculates the FFT of the current waveform data.
        _get_current_unbalance: Calculates the current unbalance from the FFT of the current waveform data.
        _get_rms_current: Calculates the RMS current from the current waveform data.
        _get_harmonic_magnitudes: Calculates the magnitudes of the harmonic components from the FFT of the current waveform data.
        _get_thd: Calculates the total harmonic distortion (THD) from the FFT of the current waveform data.
        _extract_features: Calculates the feature vector from the current waveform data.

    """

    def __init__(self, current_data, fundamental_frequency=50, sampling_rate=10000):
        if not isinstance(current_data, pd.DataFrame):
            raise TypeError("current_data must be a numpy array.")
        if len(current_data.shape) != 2:
            raise ValueError("current_data must be a 2D array.")

        self.current_data = current_data
        self.sampling_rate = sampling_rate
        self.fundamental_frequency = fundamental_frequency
        self.fft_data = self._get_fft(self.current_data)
        self.feature_dictionary = self._extract_features(
            self.current_data, self.fft_data, sampling_rate, fundamental_frequency)

    def _get_fft(self, df):
        """
        Calculates the FFT of the current waveform data.

        Parameters:
            df (pandas.DataFrame): The current waveform data as a pandas DataFrame.

        Returns:
            numpy.ndarray: The FFT of the current waveform data.
        """
        return np.fft.fft(df, axis=0)

    def _get_current_unbalance(self, fft_data: np.ndarray) -> float:
        """
        Calculates the current unbalance from the FFT of the current waveform data.

        Parameters:
            fft_data (numpy.ndarray): The FFT of the current waveform data.

        Returns:
            float: The current unbalance.
        """
        phase_angles = np.angle(fft_data, deg=True)
        phase_angles_diff = np.diff(phase_angles, axis=0)
        current_unbalance = np.max(np.abs(phase_angles_diff))
        return current_unbalance

    def _get_rms_current(self, df: np.ndarray) -> float:
        """
        Calculates the RMS current from the current waveform data.

        Parameters:
            df (pandas.DataFrame): The current waveform data as a pandas DataFrame.

        Returns:
            float: The RMS current.
        """
        rms_current = np.sqrt(np.mean(df**2))
        return rms_current

    def _get_harmonic_magnitudes(self, fft_data: np.ndarray, fs=10000, ffreq=50):
        """
        Calculates the magnitudes of the harmonic components from the FFT of the current waveform data.

        Parameters:
            fft_data (numpy.ndarray): The FFT of the current waveform data.
            fs (float): The sampling frequency of the current waveform data.
            ffreq (float): The fundamental frequency of the current waveform.

        Returns:
            numpy.ndarray: The magnitudes of the harmonic components.
        """
        fundamental_freq = ffreq  # or 60 Hz depending on the AC power supply frequency
        harmonic_freqs = [2*fundamental_freq, 3*fundamental_freq,
                          4*fundamental_freq]  # extract 2nd, 3rd and 4th harmonics
        # compute indices of harmonic components
        harmonic_indices = [int(freq/fs * (len(fft_data)/2))
                            for freq in harmonic_freqs]

        # extract magnitudes of harmonic components
        harmonic_magnitudes = np.abs(fft_data[harmonic_indices])
        return harmonic_magnitudes

    def _get_thd(self, fft_data: np.ndarray, sampling_rate: int, fundamental_frequency: int) -> float:
        """
        Calculates the total harmonic distortion (THD) of the current data.

        Parameters:
            fft_data (numpy.ndarray): The FFT of the current data.
            sampling_rate (int): The sampling rate of the current data 
            fundamental_frequency (int): The fundamental frequency of the current waveform.

        Returns:
        --------
        thd: float
            The THD value.
        """
        fundamental_index = int(
            fundamental_frequency * (len(fft_data)/2) / sampling_rate)
        thd = np.sqrt(np.sum(np.abs(fft_data[fundamental_index-2:fundamental_index+3])**2) - np.abs(
            fft_data[fundamental_index])**2) / np.abs(fft_data[fundamental_index])
        return thd

    def _extract_features(self, current_data: np.ndarray, fft_data: np.ndarray, sampling_rate: int, fundamental_frequency: int) -> Dict[str, object]:
        """
        Computes and returns a set of electrical features.

        Args:
        - current_data: A numpy array containing the raw current data.
        - fft_data: A numpy array containing the FFT of the current data.
        - sampling_rate: The sampling rate used to collect the current data.
        - fundamental_frequency: The fundamental frequency of the AC power supply.

        Returns:
        - A tuple containing the following electrical features:
            - fft_data: The FFT of the current data.
            - harmonic_magnitudes: The magnitudes of the 2nd, 3rd, and 4th harmonics.
            - thd: The total harmonic distortion of the current.
            - rms_current: The root mean square current.
            - current_unbalance: The maximum phase angle difference between any two phases.

        Raises:
        - ValueError: If the input current data has a different shape than expected.
        """
        if len(current_data.shape) != 2:
            raise ValueError(
                "Input current data must have shape (n_samples, n_channels).")
        # Compute the magnitudes of the 2nd, 3rd, and 4th harmonics.
        harmonic_magnitudes = self._get_harmonic_magnitudes(
            fft_data, sampling_rate, fundamental_frequency)
        thd = self._get_thd(fft_data, sampling_rate, fundamental_frequency)
        rms_current = self._get_rms_current(current_data)
        current_unbalance = self._get_current_unbalance(fft_data=fft_data)
        # Return a tuple containing the computed electrical features.
        return {'fft_data': fft_data, 'harmonic_magnitudes': harmonic_magnitudes, 'thd': thd, 'rms_current': rms_current, 'current_unbalance': current_unbalance}


# class AnomalyDetector:
    
#     def DecisionTreeClassifier():

#         # Create a dataset with features and labels
#         X = np.concatenate((harmonic_magnitudes, current_unbalance, thd, rms_current), axis=1)
#         y = np.zeros(len(X))  # initialize all labels to 0 (normal)

#         # Set the threshold for anomaly detection
#         threshold = 0.2

#         # Determine the labels for anomalous data
#         for i in range(len(X)):
#             for j in range(X.shape[1]):
#                 if abs(X[i][j] - benchmark_data[j]) > threshold:
#                     y[i] = 1  # set label to 1 (anomalous)

#         # Train a decision tree classifier
#         clf = DecisionTreeClassifier()
#         clf.fit(X, y)

#         # Use the classifier to predict labels for new data
#         # new_fft,hm,nthd,rms,ubal
#         X_new = np.concatenate((hm, ubal, nthd, rms), axis=1)
#         y_pred = clf.predict(X_new)

#         # Print the predicted labels
#         print("Predicted labels:", y_pred)