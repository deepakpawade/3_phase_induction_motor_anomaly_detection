import os
import time
from tqdm import tqdm
class Sender:
    def __init__(self, receiver):
        self.receiver = receiver
        self.sent_files = set()
        self.found_new_file = False


    def scan_directory(self, directory):
        while True:
            file_list = os.listdir(directory)
            diff = len(file_list) - len(self.sent_files)
            if diff:
                self.found_new_file = True
                print(f'found {diff} files')
                for filename in tqdm (file_list) :
                    filepath = os.path.join(directory, filename) 
                    if os.path.isfile(filepath) and filename not in self.sent_files :
                        self.send_file(filepath)
                        self.sent_files.add(filename)
            else:
                if self.found_new_file:
                    print("No new files found. Waiting...")
                self.found_new_file = False
            time.sleep(1)
            
    def send_file(self, filepath):
        while not self.receiver.ready():
            time.sleep(1)
        with open(filepath, 'rb') as f:
            self.receiver.receive_file(f, os.path.basename(filepath))
