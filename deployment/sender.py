import os
import time

class Sender:
    def __init__(self, receiver):
        self.receiver = receiver
        
    def scan_directory(self, directory):
        while True:
            for filename in os.listdir(directory):
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    self.send_file(filepath, filename)
            time.sleep(1)
            
    def send_file(self, filepath, filename):
        while not self.receiver.ready():
            time.sleep(1)
        with open(filepath, 'rb') as f:
            self.receiver.receive_file(filename, f.read())
