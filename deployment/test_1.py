from sklearn.mixture import GaussianMixture
import numpy as np
import copy
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./induction_motor_anomaly_detection/')
import modules,scaler
import pickle

    
# gmm = pickle.load(open('./deployment/models/gmm_raw', 'rb'))   
# normal_data = pd.read_csv('./data/combined_data.csv')

# train_scaler = scaler.Scaler()
# train_scaler.fit_unlabelled_data(normal_data)

# train_scaled = train_scaler.transform(normal_data)
# # anomalous_data = pd.read_csv('../anomalous_data/combined_anomalous.csv')

# print(gmm.predict_anomalies(train_scaled[0:10000]))
directory= './data/s3Files/'
for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    print(filename)