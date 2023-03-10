# 3 phase induction motor anomaly detection
  
  ## Description 
  See `approaches` folder to see specific and working as well as on going implementations
  Below are generic modules implemented. (PS :  This ReadMe may not be upto date)
  
  The data set contains current readings of a 3-phase AC motor (3.2hp).

  ## Table of Contents
  * [Dataset](#dataset)
  * [Modules](#modules)
 
  
  ## Dataset
  
  
    Each file has 10000 3-phase current readings. 
    There are 317 files, which means the readings were taken for 317 seconds. 
    There are 3 columns in each file represent the 3 currents.
    Based on the information provided, it seems unlikely that the frequency of the 
    current supplied is 10K Hz, as this is much higher than the typical frequency of AC power in most regions,
    which is usually around 50-60 Hz. It's possible that the data is being sampled at a rate 
    of 10K-15K Hz, but this would not necessarily reflect the frequency of the current itself.
  
  ## Modules 



Class ElectricalFeatureExtractor


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



Class AnomalyDetector


    Anomaly detection class that uses three different algorithms to detect anomalies in test data.
    IsolationForest,LocalOutlierFactor,OneClassSVM,GMM as well as mahalanobis_distance
    Attributes:
        None

    Methods:
        get_predictions: Takes in normal data and test data and uses three different algorithms to detect
            anomalies in the test data. Returns the number of anomalies detected by each algorithm.
        mahalanobis_distance: Calculates the Mahalanobis distance between normal data features and test data features and flags any test data feature vectors with a distance greater than a specified threshold as anomalous.



Class AnomalyGenerator


    This class generates anomalies in a given Pandas DataFrame by adding various anomalies to its data.

Class Noise Generator


    This class adds noise to the data