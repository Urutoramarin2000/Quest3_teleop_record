
import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.linalg import norm, solve

from pytransform3d import rotations
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore, Value
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from utils.vuer.TeleVision_controller_pybullet import OpenTeleVision
from realsense.realsense_tools import RealSenseTools
from utils.vuer.constants_vuer import tip_indices
from utils.msg_communicator import MsgPublisher,  MsgSubscriber
from utils.IK_solver_pybullet import robot_solver
from utils.vuer.Preprocessor_controller import VuerPreprocessor


class VuerTeleop:
    def __init__(self):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.img_shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.img_shm.buf)


        self.num_dof = 6
        self.dof_shm = shared_memory.SharedMemory(create=True, size=self.num_dof * np.float64().itemsize)
        self.dof_array = np.ndarray((self.num_dof, ), dtype=np.float64, buffer=self.dof_shm.buf)

        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.img_shm.name, self.num_dof, self.dof_shm.name, toggle_streaming,  cert_file="./cert.pem", key_file="./key.pem", static_root=Path(__file__).parent / "../assets")
        self.processor = VuerPreprocessor()
        
    def step(self):
        head_mat, right_wrist_mat, trigger = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        # left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([0.4, -0.5, 1.3]),
        #                             rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([0.4, 0.1, 0.8]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])


        return head_rmat, right_pose, right_wrist_mat, trigger

def arm_state_receiver(shm_name, num_dof, ip, port):
    try:
        arm_state_shm = shared_memory.SharedMemory(name=shm_name)
        arm_state_array = np.ndarray((num_dof,), dtype=np.float64, buffer=arm_state_shm.buf)
        arm_state_sub = MsgSubscriber(ip=ip, port=port, topic='arm_state')

        while True:
            msg_recv = arm_state_sub.recv()
            real_arm_dof = np.array(msg_recv['arm_state'], dtype=np.float32)
            np.copyto(arm_state_array, real_arm_dof[1:])

    except Exception as e:
        print(f"arm_state error: {e}")
    finally:
        arm_state_shm.close() 
        arm_state_shm.unlink()

def Teleop_controller(img = False, camera_serial = "748512060307"):
    # 748512060307 D415
    # 218622275838 D405
    teleoperator = VuerTeleop()
    ik_solver = robot_solver(robot_path='../assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf')
    cmd_pub = MsgPublisher(port=22222)
    real_arm_state = False
    cmd = np.zeros(7)
    if img:
        realsense = RealSenseTools(camera_serial=camera_serial, w=1280, h=720)    
    if real_arm_state:
        arm_state_process = Process(target=arm_state_receiver, args=(teleoperator.dof_shm.name, teleoperator.num_dof, '192.168.1.3', 22222))
        arm_state_process.daemon = True # 主进程退出时子进程也退出
        arm_state_process.start()

    try:
        while True:
            head_rmat, right_pose, right_pose_mat, button= teleoperator.step()
            if button[1]:

                if button[0]:
                    griper_width = 0.02  # 最小宽度（闭合状态）
                else:
                    griper_width = 0.11  # 夹爪打开
                griper_cmd = (np.clip(griper_width, 0.02, 0.11) - 0.02) / 0.09 * 100
                cmd[0] = griper_cmd

                # 旋转操作+坐标offset
                target_pos = right_pose[:3] + [-0.25, 0, -1.2]
                print(target_pos)
                target_rotation = R.from_quat(right_pose[3:]).as_matrix() # 将四元数转换为旋转矩阵
                rotation_x = R.from_euler('x', -90, degrees=True).as_matrix() # 定义绕 X 轴旋转 -90 度的旋转矩阵
                rotation_y = R.from_euler('y', 90, degrees=True).as_matrix() # 定义绕 y 轴旋转 90 度的旋转矩阵
                target_rotation_transformed = target_rotation @ rotation_x @ rotation_y   
                target_rotation_quat = R.from_matrix(target_rotation_transformed).as_quat()

                new_dof_pos = ik_solver.solve_ik_quat(target_pos, target_rotation_quat)
                # new_dof_pos = ik_solver.solve_ik_quat(right_pose[:3], right_pose[3:])
                cmd[1:] = new_dof_pos
                msg={'action': list(cmd)}
                cmd_pub.send(topic='arm_action', msg=msg)
                print(msg)
                if real_arm_state:  
                    pass
                else:
                    np.copyto(teleoperator.dof_array[:], new_dof_pos)       
            if button[2]:
                cmd = [100.0, 0.09588302035969265, -2.0823024194051465, 2.611052835276107, -0.06805776867642338, 0.5954223615574694, 0.03230260455874678]
                msg = {'action': list(cmd)}
                cmd_pub.send(topic='arm_action', msg=msg)
                print("回到零位", msg) 
                np.copyto(teleoperator.dof_array[:], cmd[1:])
            if img:
                color_img, depth_img, depth_frame, depth_colormap = realsense.recv()
                color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
                np.copyto(teleoperator.img_array, np.hstack((color_img, color_img)))

    except KeyboardInterrupt:
        exit(0)


if __name__ == '__main__':
    Teleop_controller(img=True)