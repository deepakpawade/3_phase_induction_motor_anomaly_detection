import pandas as pd
import time
from threading import Lock
import warnings
warnings.filterwarnings("ignore")
import sys
# sys.path.append('./induction_motor_anomaly_detection/')
sys.path.append('./')
import modules,scaler

class Receiver:
    def __init__(self, model, log_path):
        self.model = model
        self.lock = Lock()
        self.buffer = []
        self.logfile = log_path
        
    def receive_file(self, filedata, filename):
        with self.lock:
            self.buffer.append((filedata, filename))
            
    def process_files(self, stop_event):
        while not stop_event.is_set():
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
