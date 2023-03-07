# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import modules
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# create a combined normal data
# datafolder = '/data/s3Files/'
# data_files = os.listdir(datafolder)
# normal_data= pd.DataFrame()
# for file in data_files:
#     data = pd.concat([normal_data,pd.read_csv(datafolder+file,header=None,sep=',')])
# normal_data.drop(columns=[0,4],inplace=True)
# normal_data.dropna(inplace=True)
# normal_data.columns = ['current_1', 'current_2', 'current_3']
# normal_data.reset_index(drop=True,inplace=True)
normal_data = pd.read_csv('./data/combined_data.csv')

# features that can be engineered 
# data['BrokenRotorBars'] = abs(data['current_1']) - abs(data['current_2'])
# data['BearingFault'] = abs(data['current_2']) - abs(data['current_3'])
# data['Eccentricity'] = abs(data['current_3']) - abs(data['current_1'])
# normal_data.to_csv('./data/combined_data.csv', index=False)

# creating anomalous data
# anomalous = modules.AnomalyGenerator()
# anomalous_data = anomalous.add_anomalies(normal_data,number_of_anomalies=1000,data_length=1000)
# anomalous_data.to_csv('anomalous_data.csv', index=False)
anomalous_data = pd.read_csv('./anomalous_data/anomalous_data.csv')


benchmark_features = modules.ElectricalFeatureExtractor(current_data=normal_data)
test_features = modules.ElectricalFeatureExtractor(current_data=anomalous_data)
detector = modules.AnomalyDetector
detector.mahalanobis_distance(benchmark_features.feature_dictionary,test_features.feature_dictionary)
detector.get_predictions(normal_data,anomalous_data)