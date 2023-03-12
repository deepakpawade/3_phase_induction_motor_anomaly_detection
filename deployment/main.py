from sender import Sender
from receiver import Receiver
import threading
import pickle


gmm = pickle.load(open('./deployment/models/gmm_raw', 'rb')) 

receiver = Receiver(gmm)
sender = Sender(receiver)
t1 = threading.Thread(target=sender.scan_directory, args=('./data/s3Files/',))
t2 = threading.Thread(target=receiver.process_files)

t1.start()
t2.start()

t1.join()
t2.join()
