from sender import Sender
from receiver import Receiver
import threading
import pickle
import argparse
import signal
import sys



def signal_handler(signal, frame):
    print('You pressed Ctrl+C!. EXITING')
    sys.exit(0)

def main(directory, log_file):
    stop_event = threading.Event()
    signal.signal(signal.SIGINT, signal_handler)   

    gmm = pickle.load(open('./models/gmm_raw', 'rb')) 

    receiver = Receiver(gmm,log_file)
    sender = Sender(receiver)
    t1 = threading.Thread(target=sender.scan_directory, args=(directory, stop_event))
    t2 = threading.Thread(target=receiver.process_files,args=(stop_event,))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Induction Motor Anomaly Detector")
    parser.add_argument('--directory', type=str, help='Directory to scanning folder')
    parser.add_argument('--log_file', type=str, help='Directory to store log file')
    args = parser.parse_args()
    main(args.directory,args.log_file)
