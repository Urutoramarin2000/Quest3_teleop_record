import time
from vuer import Vuer
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene, Urdf, Movable
from vuer.schemas import Sphere # 只导入 Sphere 组件
from vuer.schemas import MotionControllers
from pathlib import Path
from multiprocessing import Array, Process, shared_memory, Manager, Event, Semaphore
import numpy as np
import asyncio
import matplotlib.pyplot as plt
from webrtc.zed_server import *
from realsense.realsense_tools import RealSenseTools
from threading import Thread
import pyrealsense2 as rs
import cv2

class OpenTeleVision:
    def __init__(self, img_shape, img_shm_name, num_dof, dof_shm_name, toggle_streaming, stream_mode="image", cert_file="./cert.pem", key_file="./key.pem", ngrok=False):
        # self.app=Vuer()
        self.img_shape = img_shape
        # self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        # self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3, static_root=Path(__file__).parent / "../assets")
        self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, static_root=Path(__file__).parent / "../assets")
        
        # print("----------------", Path(__file__).parent / "../assets")
        # /home/cowa/Documents/TeleVision/assets/cowa_legged_wheel_arm/urdf
        self.app.add_handler("CONTROLLER_MOVE")(self.on_controller_move)
        self.app.add_handler("CAMERA_MOVE")(self.on_cam_move)

        image_shm = shared_memory.SharedMemory(name=img_shm_name)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=image_shm.buf)

        dof_shm = shared_memory.SharedMemory(name=dof_shm_name)
        self.dof_array = np.ndarray((num_dof, ), dtype=np.float64, buffer=dof_shm.buf)
        
        self.dof_pos = np.ndarray(6)
        print("image shape",self.img_array.shape)
        self.app.spawn(start=False)(self.main_image)

        self.left_controller_shared = Array('d', 16, lock=True)
        self.right_controller_shared = Array('d', 16, lock=True)
        self.right_trigger_shared = Value('b', False)
        
        self.head_matrix_shared = Array('d', 16, lock=True)
        self.aspect_shared = Value('d', 1.0, lock=True)

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
        session.set @ DefaultScene(
            grid=False, #添加地面 
            children=[  
                # Movable( # 可移动的机器人模型
                    MotionControllers(stream=True, key="motion-controller"),
                    Urdf(
                        src="https://localhost:8012/static/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf",
                        jointValues={
                            "joint15": -0.2,
                            "joint16": -0.2,
                            "joint17": 0.2,
                            "joint18": 0.2,
                            "joint19": -0.25 * 3.14,
                            "joint20": -0.25 * 3.14,
                        },
                        key="robot",
                        # x, z , y
                        position=[0, 0, -1],
                        rotation=[-3.14 / 2, 0, 3.14 / 2],
                        scale=1,
                    ),
                    
                    # Sphere(
                    #     radius=1,
                    #     color="red",
                    #     key="simple_shape"
                    # ),
                    # position=[0, 0, 0],
                    # scale=0.5,
                
                ImageBackground(     # 背景图像
                    # display_image,
                    np.zeros((480, 640, 3), dtype=np.uint8),
                    format="jpeg",
                    quality=80,
                    key="background",
                    interpolate=True,
                    fixed=True,
                    aspect=1.5,
                    # distanceToCamera=0.5,
                    height = 3,
                    position=[0, 1, -3],
                    rotation=[0, 0, 0],
                    layers=1, 
                    alphaSrc="./vinette.jpg"
                ),
            ]
        )

        while True:
            display_image = self.img_array
            # updated_pos = self.dof_pos
            updated_pos = self.dof_array
            # print("---------dof:",self.dof_array)

            # print("display_img_shape:", display_image.shape, "updated_pos:", updated_pos)

            session.update @ Urdf(
                jointValues={
                    "joint15": updated_pos[0],
                    "joint16": updated_pos[1],
                    "joint17": updated_pos[2],
                    "joint18": updated_pos[3],
                    "joint19": updated_pos[4],
                    "joint20": updated_pos[5],
                    # "joint15": 0,
                    # "joint16": 0,
                    # "joint17": 0,
                    # "joint18": 0,
                    # "joint19": 0,
                    # "joint20": 0,
                },
                key="robot",
                # position=[0, 0, 0],
                # scale=1,
            )

            session.upsert @ ImageBackground(
                # Can scale the images down.
                display_image,
                format="jpeg",
                quality=80,
                key="background",
                interpolate=True,
                fixed=True,
                aspect=1.5,
                # distanceToCamera=0.5,
                height = 3,
                position=[0, 1, -3],
                # rotation=[0, 0, 0],
                layers=1, 
                # alphaSrc="./vinette.jpg"
            )
            
            
            await asyncio.sleep(0.03)

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



    
if __name__ == "__main__":
    resolution = (720, 1280)
    crop_size_w = 340  # (resolution[1] - resolution[0]) // 2
    crop_size_h = 270
    resolution_cropped = (resolution[0] - crop_size_h, resolution[1] - 2 * crop_size_w)  # 450 * 600
    img_shape = (2 * resolution_cropped[0], resolution_cropped[1], 3)  # 900 * 600
    img_height, img_width = resolution_cropped[:2]  # 450 * 600
    shm = shared_memory.SharedMemory(create=True, size=np.prod(img_shape) * np.uint8().itemsize)
    shm_name = shm.name
    img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shm.buf)

    tv = OpenTeleVision(resolution_cropped, cert_file="../cert.pem", key_file="../key.pem")
    while True:
        # print(tv.left_landmarks)
        # print(tv.left_hand)
        # tv.modify_shared_image(random=True)
        time.sleep(1)
