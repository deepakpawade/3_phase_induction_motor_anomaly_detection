import numpy as np
import copy
import os
import pandas as pd
from typing import Dict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import mahalanobis

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
        self.feature_dataframe = self._get_features(self.current_data)


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

    def _get_harmonic_magnitudes(self, fft_data: np.ndarray, fs=10000, ffreq=50, normalized = False):
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

        if normalized:
        # normalize harmonic magnitudes using z-score normalization
            mean = np.mean(harmonic_magnitudes)
            std = np.std(harmonic_magnitudes)
            harmonic_magnitudes = (harmonic_magnitudes - mean) / std

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

    def _get_features(self,df):
        """
        Computes various features of the input DataFrame using the following steps:
        1. Groups the data into segments of 10000 data points and computes the mean, variance, and standard deviation of each segment
        2. Computes the harmonic magnitudes, total harmonic distortion, root-mean-square current, and current unbalance for each segment using the Fast Fourier Transform (FFT)
        3. Adds the computed features to the DataFrame

        Args:
        df (pandas.DataFrame): Input DataFrame of power consumption data

        Returns:
        list: A list containing the DataFrame of computed features and the FFT data
        """
        
        features = df.groupby(df.index // 10000).agg(['mean', 'var', 'std'])
        mags = []
        thds = []
        rmss = []
        unbals = []
        fft_data = np.fft.fft(df)
        for i in range(0,df.shape[0],10000):
            mags.append(self._get_harmonic_magnitudes(fft_data[i:i+10000]))
            thds.append(self._get_thd(fft_data[i:i+10000], 10000, 50))
            rmss.append(self._get_rms_current(df[i:i+10000]))
            unbals.append(self._get_current_unbalance(fft_data[i:i+10000]))

            
        for i, row in features.iterrows():
            harmonic_magnitudes = mags[i]
            thd = thds[i]
            rms = rmss[i]
            unbal = unbals[i]
            features.loc[i, ('current_1', 'harmonic_2')] = harmonic_magnitudes[0][0]
            features.loc[i, ('current_1', 'harmonic_3')] = harmonic_magnitudes[0][1]
            features.loc[i, ('current_1', 'harmonic_4')] = harmonic_magnitudes[0][2]
            features.loc[i, ('current_2', 'harmonic_2')] = harmonic_magnitudes[1][0]
            features.loc[i, ('current_2', 'harmonic_3')] = harmonic_magnitudes[1][1]
            features.loc[i, ('current_2', 'harmonic_4')] = harmonic_magnitudes[1][2]
            features.loc[i, ('current_3', 'harmonic_2')] = harmonic_magnitudes[2][0]
            features.loc[i, ('current_3', 'harmonic_3')] = harmonic_magnitudes[2][1]
            features.loc[i, ('current_3', 'harmonic_4')] = harmonic_magnitudes[2][2]

            features.loc[i, ('current_1', 'thd')] = thd[0]
            features.loc[i, ('current_2', 'thd')] = thd[1]
            features.loc[i, ('current_3', 'thd')] = thd[2]

            features.loc[i, ('current_1', 'rms')] = rms[0]
            features.loc[i, ('current_2', 'rms')] = rms[1]
            features.loc[i, ('current_3', 'rms')] = rms[2]
            features.loc[i, ('current_unbalance', '')] = unbal
        features.columns = features.columns.to_flat_index()
        features.columns = ['_'.join(column) for column in features.columns]
        return [features, fft_data]


class AnomalyDetector:
    """
    Anomaly detection class that uses three different algorithms to detect anomalies in test data.

    Attributes:
        None

    Methods:
        get_predictions: Takes in normal data and test data and uses three different algorithms to detect
            anomalies in the test data. Returns the number of anomalies detected by each algorithm.
        mahalanobis_distance: Calculates the Mahalanobis distance between normal data features and test data
            features and flags any test data feature vectors with a distance greater than a specified threshold
            as anomalous.
    """
    
    def GaussianMixture(train_feature_dataframe, test_feature_dataframe ,n_components=1, covariance_type='full', k_threshold_std_dev = 3):
        """
        Perform anomaly detection using Gaussian Mixture Model.
        Parameters:
        train_feature_dataframe (pandas.DataFrame): Dataframe containing training data features
        test_feature_dataframe (pandas.DataFrame): Dataframe containing test data features
        n_components (int): Number of Gaussian distributions to fit
        covariance_type (str): Type of covariance matrix to use, can be 'full', 'tied', 'diag', 'spherical'
        k_threshold_std_dev (float): Number of standard deviations below the mean to set the anomaly threshold

        Returns:
        None, but prints 'anomalous' or 'not anomalous' depending on whether the test data is considered an anomaly.
        """
        gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
        gmm.fit(train_feature_dataframe)
        scores = gmm.score_samples(train_feature_dataframe)
        k =  k_threshold_std_dev
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        threshold = mean_score - k * std_score
        is_anomaly = gmm.score_samples(test_feature_dataframe) < threshold
        if is_anomaly:
            print('anomalous')
        else:
            print('not anomalous')



    def get_predictions(normal_data, test_data):
        """
        Use three different algorithms to detect anomalies in test data.

        Args:
            normal_data: A numpy array containing the normal data used to train the algorithms.
            test_data: A numpy array containing the test data to be analyzed for anomalies.

        Returns:
            A tuple containing the number of anomalies detected by each algorithm.
        """
        # Initialize anomaly detection algorithms
        iforest = IsolationForest(n_estimators=100, contamination=0.05)
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
        ocsvm = OneClassSVM(kernel='rbf', nu=0.05)

        # Initialize progress bars for fitting
        iforest_fit_bar = tqdm(total=3, desc="Fitting Isolation Forest")
        lof_fit_bar = tqdm(total=3, desc="Fitting Local Outlier Factor")
        ocsvm_fit_bar = tqdm(total=3, desc="Fitting One-Class SVM")

        # Fit models on normal data
        iforest.fit(normal_data)
        iforest_fit_bar.update(1)
        lof.fit(normal_data)
        lof_fit_bar.update(1)
        ocsvm.fit(normal_data)
        ocsvm_fit_bar.update(1)

        # Close fitting progress bars
        iforest_fit_bar.close()
        lof_fit_bar.close()
        ocsvm_fit_bar.close()

        # Initialize progress bars for predicting
        iforest_pred_bar = tqdm(total=len(test_data), desc="Isolation Forest")
        lof_pred_bar = tqdm(total=len(test_data), desc="Local Outlier Factor")
        ocsvm_pred_bar = tqdm(total=len(test_data), desc="One-Class SVM")

        # Predict anomalies in test data
        iforest_preds = []
        lof_preds = []
        ocsvm_preds = []
        for i, data_point in enumerate(test_data):
            iforest_pred = iforest.predict([data_point])[0]
            lof_pred = lof.predict([data_point])[0]
            ocsvm_pred = ocsvm.predict([data_point])[0]
            iforest_preds.append(iforest_pred)
            lof_preds.append(lof_pred)
            ocsvm_preds.append(ocsvm_pred)
            iforest_pred_bar.update(1)
            lof_pred_bar.update(1)
            ocsvm_pred_bar.update(1)

        # Close predicting progress bars
        iforest_pred_bar.close()
        lof_pred_bar.close()
        ocsvm_pred_bar.close()

        # Print number of anomalies detected by each algorithm
        print("Isolation Forest: ", np.count_nonzero(iforest_preds == -1))
        print("Local Outlier Factor: ", np.count_nonzero(lof_preds == -1))
        print("One-Class SVM: ", np.count_nonzero(ocsvm_preds == -1))




    def mahalanobis_distance(normal_features, test_features, threshold=3.0):
        """
        Computes the Mahalanobis distance between each test data feature vector and the normal mean vector, and flags any
        test data feature vectors that have a Mahalanobis distance greater than the threshold as anomalous.

        Args:
            normal_features (dict): A dictionary containing the normal data features with keys 'harmonic_magnitudes', 
                                    'thd', 'rms_current', and 'current_unbalance'.
            test_features (dict): A dictionary containing the test data features with keys 'harmonic_magnitudes', 'thd', 
                                'rms_current', and 'current_unbalance'.
            threshold (float): The threshold distance value based on the desired level of anomaly detection sensitivity. 
                            Defaults to 3.0.

        Returns:
            None: The function prints a message indicating whether any anomalous feature vectors were detected or not.

        """
        
        # Extract normal and test data features
        normal_data = {
            'harmonic_magnitudes': normal_features['harmonic_magnitudes'],
            'thd': normal_features['thd'],
            'rms_current': normal_features['rms_current'].values,
            'current_unbalance': normal_features['current_unbalance']
        }
        
        test_data = {
            'harmonic_magnitudes': test_features['harmonic_magnitudes'],
            'thd': test_features['thd'],
            'rms_current': test_features['rms_current'].values,
            'current_unbalance': test_features['current_unbalance']
        }

        # Compute mean and covariance matrix of normal data features
        normal_mean = np.mean(normal_data['harmonic_magnitudes'], axis=0)
        normal_cov = np.cov(normal_data['harmonic_magnitudes'].T)

        # Compute the Mahalanobis distance between each test data feature vector and the normal mean vector
        distances = []
        for i in range(test_data['harmonic_magnitudes'].shape[0]):
            distance = mahalanobis(test_data['harmonic_magnitudes'][i], normal_mean, np.linalg.inv(normal_cov))
            distances.append(distance)

        # Flag any test data feature vectors that have a Mahalanobis distance greater than the threshold as anomalous
        anomalous_indices = np.where(np.array(distances) > threshold)[0]
        if anomalous_indices.size > 0:
            print("Anomalous feature vectors detected at indices:", anomalous_indices)
        else:
            print("No anomalous feature vectors detected.")

class AnomalyGenerator:
    """
    This class generates anomalies in a given Pandas DataFrame by adding noise to its data.

    Parameters:
    -----------
    None

    Methods:
    --------
    add_anomalies(dataframe, number_of_anomalies=0, data_length=0)
        Adds anomalies to the given DataFrame and returns a new DataFrame with added noise.

        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The DataFrame to which anomalies should be added.

        number_of_anomalies : int (default: 0)
            The number of anomalies to add. If 0, it adds anomalies to the entire DataFrame.

        data_length : int (default: 0)
            The length of data to which anomalies should be added. If 0, it adds anomalies to the entire DataFrame.

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with added noise, representing the anomalous data.
    """


    def add_anomalies_amplitude(self, dataframe, number_of_anomalies=0, data_length=0):
        """
        Adds anomalies to the given DataFrame and returns a new DataFrame with added noise.

        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The DataFrame to which anomalies should be added.

        number_of_anomalies : int (default: 0)
            The number of anomalies to add. If 0, it adds anomalies to the entire DataFrame.

        data_length : int (default: 0)
            The length of data to which anomalies should be added. If 0, it adds anomalies to the entire DataFrame.

        Returns:
        --------
        pandas.DataFrame
            A new DataFrame with added noise, representing the anomalous data.
        """
        

        # Determine the upper bound of the data to which anomalies should be added.
        upper_bound = number_of_anomalies if number_of_anomalies else len(dataframe)
        
        # Determine the length of the data to which anomalies should be added.
        length = data_length if data_length else len(dataframe)

        # Make a deep copy of the data up to the upper bound and length.
        df = copy.deepcopy(dataframe[0:length])
        noise_df = df.iloc[:upper_bound,:]

        # Calculate the noise amplitude based on the signal amplitude.
        noise_amplitude = 0.1 * np.max(np.abs(df))

        # Add noise to the signal using amplitude scaling.
        noise = np.random.normal(0, noise_amplitude, noise_df.shape)
        
        df.iloc[:upper_bound,:] += noise

        return df
    
    def add_noise_awgn():
        # For electric signals, one way to add noise while preserving the signal 
        # characteristics is by using a technique called additive white Gaussian noise (AWGN). 
        # This technique adds random Gaussian noise to the signal while preserving the 
        # frequency characteristics of the signal.
        # Load the signal
        signal = np.load('signal.npy')

        # Calculate the power of the signal
        signal_power = np.var(signal)

        # Set the desired SNR in dB
        snr_db = 10

        # Calculate the noise power based on the desired SNR and the signal power
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate random Gaussian noise with zero mean and the calculated noise power
        noise = np.random.normal(0, np.sqrt(noise_power), size=signal.shape)

        # Add the noise to the signal
        noisy_signal = signal + noise
        return noisy_signal


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

