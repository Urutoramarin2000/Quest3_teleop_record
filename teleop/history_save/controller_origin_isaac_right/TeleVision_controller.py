import time
from vuer import Vuer
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene
from vuer.schemas import MotionControllers
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from webrtc.zed_server import *
from realsense.realsense_tools import RealSenseTools
from threading import Thread
import pyrealsense2 as rs
import cv2

class OpenTeleVision:
    def __init__(self, img_shape, shm_name, queue, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False):
        # self.app=Vuer()
        self.img_shape = [480, 640]
        # self.img_height, self.img_width = [640,480]
        
        # plt.ion()  
        # self.fig, ax = plt.subplots()
        # self.im = ax.imshow(np.random.rand(840, 480, 3)) 

        if ngrok:
            self.app = Vuer(host='0.0.0.0', queries=dict(grid=False), queue_len=3)
        else:
            self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)

        self.app.add_handler("CONTROLLER_MOVE")(self.on_controller_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)
        # if stream_mode == "image":
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
        print("image shape",self.img_array.shape)
        self.app.spawn(start=False)(self.main_image)

        self.left_controller_shared = Array('d', 16, lock=True)
        self.right_controller_shared = Array('d', 16, lock=True)
        self.right_trigger_shared = Value('b', False)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

        # self.realsense_process = Process(target=self.realsense_frame_process, args=(shm_name, img_shape)) #  创建进程，目标函数为 realsense_frame_process
        # self.realsense_process.daemon = True
        # self.realsense_process.start()

        self.process = Process(target=self.run)
        self.process.daemon = True
        self.process.start()

    
    def run(self):
        self.app.run()
    
    async def on_cam_move(self, event, session, fps=60):
        try:
            self.head_matrix_shared[:] = event.value["camera"]["matrix"]
            self.aspect_shared.value = event.value['camera']['aspect']
        except:
            pass

    async def on_controller_move(self, event, session, fps=60):
        try:
            with self.right_controller_shared.get_lock():
                self.right_controller_shared[:] = event.value["right"]
            
            right_state = event.value.get("rightState", {})
            trigger_value = right_state.get("trigger", False) 
            a_buttom = right_state.get("aButtonValue",False)
            print('trigger_value:',trigger_value, ';a_buttom:', a_buttom)
            self.right_trigger_shared.value = trigger_value
        except: 
            pass

    async def main_image(self, session, fps=60):
        # session.upsert @ Hands(fps=fps, stream=True, key="hands", showLeft=False, showRight=False)
        session.upsert @ MotionControllers(stream=True, key="motion-controller")
        end_time = time.time()
        while True:
            start = time.time()
            print("!!!!!!!!!!!!1")
            display_image = self.img_array
            print("display_img",display_image.shape)
            # frames_2 = self.pipeline_2.wait_for_frames()
            # color_frame_2 = frames_2.get_color_frame()
            # frame_2 = np.asanyarray(color_frame_2.get_data())
            # session.upsert(
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[:self.img_height],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="left-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.778,
            #     distanceToCamera=2,
            #     position=[0, -0.5, -2],
            #     rotation=[0, 0, 0],
            # ),
            # to="bgChildren",
            # )
            session.upsert(
            ImageBackground(
                # Can scale the images down.
                display_image,
                format="jpeg",
                quality=80,
                key="background",
                interpolate=True,
                # fixed=True,
                aspect=1.66667,
                # distanceToCamera=0.5,
                height = 8,
                position=[0, -1, 3],
                # rotation=[0, 0, 0],
                layers=1, 
                alphaSrc="./vinette.jpg"
            ),
            # ImageBackground(
            #     # Can scale the images down.
            #     display_image[::2, self.img_width:],
            #     # display_image[self.img_height::2, ::2],
            #     # 'jpg' encoding is significantly faster than 'png'.
            #     format="jpeg",
            #     quality=80,
            #     key="right-image",
            #     interpolate=True,
            #     # fixed=True,
            #     aspect=1.66667,
            #     # distanceToCamera=0.5,
            #     height = 8,
            #     position=[0, -1, 3],
            #     # rotation=[0, 0, 0],
            #     layers=2, 
            #     alphaSrc="./vinette.jpg"
            # )],
            to="bgChildren",
            )
            # rest_time = 1/fps - time.time() + start
            end_time = time.time()
            await asyncio.sleep(0.03)

    # @property
    # def left_hand(self):
    #     # with self.left_hand_shared.get_lock():
    #     #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
    #     return np.array(self.left_controller_shared[:]).reshape(4, 4, order="F")
        
    
    @property
    def right_hand(self):
        # with self.right_hand_shared.get_lock():
        #     return np.array(self.right_hand_shared[:]).reshape(4, 4, order="F")
        return np.array(self.right_controller_shared[:]).reshape(4, 4, order="F")
        
    
    # @property
    # def left_landmarks(self):
    #     # with self.left_landmarks_shared.get_lock():
    #     #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def right_landmarks(self):
        # with self.right_landmarks_shared.get_lock():
            # return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)

    @property
    def head_matrix(self):
        # with self.head_matrix_shared.get_lock():
        #     return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")

    @property
    def aspect(self):
        # with self.aspect_shared.get_lock():
            # return float(self.aspect_shared.value)
        return float(self.aspect_shared.value)



    
# if __name__ == "__main__":
#     resolution = (720, 1280)
#     crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
#     crop_size_h = 270
#     resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
#     img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
#     img_height, img_width = resolution_cropped[:2]  # 450 * 600
#     shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
#     shm_name = shm.name
#     img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

#     tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
#     while True:
#         # print(tv.left_landmarks)
#         # print(tv.left_hand)
#         # tv.modify_shared_image(random=True)
#         time.sleep(1)
