import threading
import time
import cv2
import numpy as np
import base64
from utils.msg_communicator import MsgSubscriber, MsgPublisher

class Viewer:
    def __init__(self, ip):
        self.camera_d405_sub = MsgSubscriber(ip=ip, port=22405, topic='camera_d405')
        self.camera_d415_sub = MsgSubscriber(ip=ip, port=22415, topic='camera_d415')
        self.d405_img = None
        self.d415_img = None

    def run_d405(self):
        while True:
            camera_d405_msg = self.camera_d405_sub.recv()
            print('Receive d405')
            t1 = time.perf_counter()
            self.d405_img = np.frombuffer(base64.b64decode(camera_d405_msg['data']), dtype=np.uint8).reshape(camera_d405_msg['shape'])
            print(f'405 decode time: {time.perf_counter()-t1}')

    def run_d415(self):
        while True:
            camera_d415_msg = self.camera_d415_sub.recv()
            print('Receive d415')
            t1 = time.perf_counter()
            self.d415_img = np.frombuffer(base64.b64decode(camera_d415_msg['data']), dtype=np.uint8).reshape(camera_d415_msg['shape'])
            print(f'415 decode time: {time.perf_counter()-t1}')

    def show(self):
        while True:
            if self.d405_img is not None:
                cv2.imshow('d405', self.d405_img)
            if self.d415_img is not None:
                cv2.imshow('d415', self.d415_img)
            if cv2.waitKey(1) == ord('q'):
                break
        print('Exit Viewer!')
        
    def run(self):
        th1 = threading.Thread(target=self.run_d405)
        th2 = threading.Thread(target=self.run_d415)
        th1.start()
        th2.start()
        self.show()
        th2.join()
        th1.join()

if __name__ == '__main__':
    viewer = Viewer(ip='192.168.2.25')
    viewer.run()


    