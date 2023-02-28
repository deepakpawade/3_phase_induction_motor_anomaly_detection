# Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import modules


datafolder = '/data/s3Files/'
data_files = os.listdir(datafolder)
data= pd.DataFrame()
for file in data_files:
    # print(file)
    data = pd.concat([data,pd.read_csv(datafolder+file,header=None,sep=',')])
data.drop(columns=[0,4],inplace=True)
data.dropna(inplace=True)
data.columns = ['current_1', 'current_2', 'current_3']


anomalous_data = pd.read_csv('anomalous_data.csv')

getfeatures = modules.ElectricalFeatureExtractor(current_data=anomalous_data)
for key,value in getfeatures.feature_dictionary.items():
    print(value)
