import pandas as pd
import time
from threading import Lock
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./induction_motor_anomaly_detection/')
import modules,scaler

class Receiver:
    def __init__(self, model):
        self.model = model
        self.lock = Lock()
        self.buffer = []
        self.logfile = './deployment/results.log'
        
    def receive_file(self, filedata, filename):
        with self.lock:
            self.buffer.append((filedata, filename))
            
    def process_files(self):
        while True:
            with self.lock:
                if self.buffer:
                    filedata, filename = self.buffer.pop(0)
                    self.process_file(filedata, filename)
                    
    def process_file(self, filedata, filename):
        # Preprocess the file data
        df = pd.read_csv(filedata.name)
        # print(df)

        train_scaler = scaler.Scaler()
        train_scaler.fit_unlabelled_data(df)
        df_scaled = train_scaler.transform(df)
        result = self.model.predict_anomalies(df_scaled)  
        
        # Log the results
        with open(self.logfile, 'a') as f:
            f.write(f"{filename}: {result}\n")
        # print(filename)
        time.sleep(1)  # Simulate processing time
        
    def ready(self):
        with self.lock:
            return not self.buffer
