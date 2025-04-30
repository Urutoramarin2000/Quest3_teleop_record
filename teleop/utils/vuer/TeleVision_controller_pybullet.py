import time
from vuer import Vuer
from vuer import Vuer, VuerSession
from vuer.events import ClientEvent
from vuer.schemas import ImageBackground, group, Hands, WebRTCStereoVideoPlane, DefaultScene, Urdf, Movable
from vuer.schemas import Sphere 
from vuer.schemas import MotionControllers
from pathlib import Path
from multiprocessing import Array, Process, shared_memory, Manager, Event, Semaphore, Value
import numpy as np
import asyncio
import matplotlib.pyplot as plt
import cv2


class OpenTeleVision:
    def __init__(self, img_shape, img_shm_name, num_dof, dof_shm_name, toggle_streaming, cert_file="./cert.pem", key_file="./key.pem", static_root=Path(__file__).parent / "../assets"):
        
        self.img_shape = img_shape
        # self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3)
        # self.app = Vuer(host='192.168.2.30', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3, static_root=static_root)
        self.app = Vuer(host='0.0.0.0', cert=cert_file, key=key_file, queries=dict(grid=False), queue_len=3, static_root=static_root)
        
        print("----------------", Path(__file__).parent / ".../assets")
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
        self.right_button_shared = Array('b', [False, False, False, False, False])
        
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
            # with self.right_controller_shared.get_lock():
            self.right_controller_shared[:] = event.value["right"]

            right_state = event.value.get("rightState", {})
            trigger_value = right_state.get("trigger", False) 
            a_button = right_state.get("aButtonValue", False)
            b_button = right_state.get("bButtonValue", False)
            squeeze_button = right_state.get("squeezeValue", False)
            thumbstick_press = right_state.get("thumbstick", False)
            self.right_button_shared[0] = trigger_value  # 触发器
            self.right_button_shared[1] = squeeze_button # 挤压按钮
            self.right_button_shared[2] = a_button      # A 按钮
            self.right_button_shared[3] = b_button      # B 按钮
            self.right_button_shared[4] = thumbstick_press
        except: 
            pass
    async def main_image(self, session, fps=60):
        session.set @ DefaultScene(
            grid=False,
            cameraPosition=[0, 0, 0], 
            cameraRotation=[0, 0, 0],   
            children=[  
                MotionControllers(stream=True, key="motion-controller", left=False, right=True),
                Urdf(
                    src="https://localhost:8012/static/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf",
                    # src="https://192.168.2.30:8012/static/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf",
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
                    position=[0, 0.4, 0],
                    rotation=[-3.14 / 2, 0, 3.14 / 2],
                    scale=1,
                ),
                # Sphere(
                #     radius=1,
                #     color="red",
                #     key="simple_shape",
                #     position=[0, 0, 0],
                #     scale=5,
                # ),    
                ImageBackground(     
                    np.zeros((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8),
                    format="jpeg",
                    quality=20,
                    key="background",
                    interpolate=True,
                    fixed=True,
                    # aspect=1.77,
                    # distanceToCamera=0.5,
                    height = 2,
                    position=[0, 1, -3],
                    rotation=[0, 0, 0],
                    layers=1, 
                    alphaSrc="./vinette.jpg"
                ),
            ]
        )

        while True:
            display_image = self.img_array
            updated_pos = self.dof_array
            # print("display_img_shape:", display_image.shape, "updated_pos:", updated_pos)

            session.update @ Urdf(
                jointValues={
                    "joint15": updated_pos[0],
                    "joint16": updated_pos[1],
                    "joint17": updated_pos[2],
                    "joint18": updated_pos[3],
                    "joint19": updated_pos[4],
                    "joint20": updated_pos[5],
                },
                key="robot",
            )

            session.upsert @ ImageBackground(
                display_image,
                format="jpeg",
                quality=20,
                key="background",
                interpolate=True,
                fixed=True,
                position=[0, 1, -3],
                layers=1, 
            )
            
            
            await asyncio.sleep(0.03)


    # @property
    # def left_hand(self):
    #     # with self.left_hand_shared.get_lock():
    #     #     return np.array(self.left_hand_shared[:]).reshape(4, 4, order="F")
    #     return np.array(self.left_controller_shared[:]).reshape(4, 4, order="F")
    # @property
    # def left_landmarks(self):
    #     # with self.left_landmarks_shared.get_lock():
    #     #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)
    #     return np.array(self.left_landmarks_shared[:]).reshape(25, 3)      
    @property
    def button(self):
        return np.array(self.right_button_shared[:])
    
    @property
    def right_hand(self):
        return np.array(self.right_controller_shared[:]).reshape(4, 4, order="F")
    
    @property
    def right_landmarks(self):
        return np.array(self.right_landmarks_shared[:]).reshape(25, 3)
    
    @property
    def head_matrix(self):
        return np.array(self.head_matrix_shared[:]).reshape(4, 4, order="F")
    
    @property
    def aspect(self):
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
