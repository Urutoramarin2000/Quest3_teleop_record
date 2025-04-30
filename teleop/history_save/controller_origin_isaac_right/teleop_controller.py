
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import matplotlib.pyplot as plt
import casadi
import cv2
import math
import numpy as np
import torch
from numpy.linalg import norm, solve

from TeleVision_controller import OpenTeleVision
from Preprocessor_controller import VuerPreprocessor
from teleop.utils.constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.spatial.transform import Rotation as R
from pinocchio import casadi as cpin                
from weighted_moving_filter import WeightedMovingFilter
import math
from teleop.utils.msg_communicator import MsgPublisher
from realsense.realsense_tools import RealSenseTools
class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (480, 640)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        



    def step(self):
        head_mat, right_wrist_mat, trigger = self.processor.process(self.tv)

        head_rmat = head_mat[:3, :3]

        # left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([0.4, -0.5, 1.3]),
        #                             rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([0.4, 0.1, 0.8]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])


        return head_rmat, right_pose, trigger

class Sim:
    def __init__(self,
                 print_freq=False):
        self.print_freq = print_freq
        # zmq 
        self.cmd_pub = MsgPublisher(port=22222)
        self.cmd = np.zeros(7)
        # initialize gym
        self.gym = gymapi.acquire_gym()

        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 120
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False

        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        # load table asset
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        # load cube asset
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        cowa_asset_path = "cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        cowa_asset = self.gym.load_asset(self.sim, asset_root, cowa_asset_path, asset_options)

        self.dof = self.gym.get_asset_dof_count(left_asset)

        self.dof_cowa = self.gym.get_asset_dof_count(cowa_asset)
        # set up the env grid
        num_envs = 1
        num_per_row = int(math.sqrt(num_envs))
        env_spacing = 1.25
        env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        # table
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(self.env, table_asset, pose, 'table', 0)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # cube
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(self.env, cube_asset, pose, 'cube', 0)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(self.env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

        # left_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.left_handle = self.gym.create_actor(self.env, left_asset, pose, 'left', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.left_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(self.env, self.left_handle, gymapi.DOMAIN_SIM)

        # cowa 
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 0)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.cowa_handle = self.gym.create_actor(self.env, cowa_asset, pose, 'cowa', 1, 1)
        self.cowa_state = np.zeros(self.dof_cowa, gymapi.DofState.dtype)
        self.cowa_state[1] = -0.5
        self.cowa_state[2] = 0.8
        self.gym.set_actor_dof_states(self.env, self.cowa_handle, self.cowa_state ,
                                      gymapi.STATE_ALL)
        cowa_idx = self.gym.get_actor_index(self.env, self.cowa_handle, gymapi.DOMAIN_SIM)

        # right_hand
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0, 1.6)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        self.right_handle = self.gym.create_actor(self.env, right_asset, pose, 'right', 1, 1)
        self.gym.set_actor_dof_states(self.env, self.right_handle, np.zeros(self.dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(self.env, self.right_handle, gymapi.DOMAIN_SIM)

        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]


        self.cowa_root_states = self.root_states[cowa_idx]
        self.world_offset = self.cowa_root_states[:3]

        # create default viewer
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            print("*** Failed to create viewer")
            quit()
        cam_pos = gymapi.Vec3(1, 1, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([0.2, 0, 0.7])

        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.left_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.left_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        self.right_camera_handle = self.gym.create_camera_sensor(self.env, camera_props)
        self.gym.set_camera_location(self.right_camera_handle,
                                     self.env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))
                                     

        ### IK Initialization
        self.robot = pin.buildModelFromUrdf("../assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf") # Directly assign robot, not robot.model
        self.robot_data = pin.Data(self.robot)

        
        self.first_compute_flag = True
        self.left_compute_flag = True
        self.right_compute_flag = True
        self.last_config = None


        # Creating Casadi models and data for symbolic computing
        self.cmodel = cpin.Model(self.robot)
        self.cdata = self.cmodel.createData()

        # Creating symbolic variables
        self.cq = casadi.SX.sym("q", self.robot.nq, 1) 
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        # Get the hand joint ID and define the error function
        self.end_effector = self.robot.getFrameId("link20")


        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.end_effector].translation - self.cTf_l[:3,3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l],
            [
                casadi.vertcat(
                    cpin.log3(self.cdata.oMf[self.end_effector].rotation @ self.cTf_l[:3,:3].T),
                )
            ],
        )

        # Defining the optimization problem
        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.robot.nq)
        self.var_q_last = self.opti.parameter(self.robot.nq)   # for smooth
        self.param_tf_l = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(self.translational_error(self.var_q, self.param_tf_l))
        self.rotation_cost = casadi.sumsqr(self.rotational_error(self.var_q, self.param_tf_l))
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        # Setting optimization constraints and goals
        self.opti.subject_to(self.opti.bounded(
            self.robot.lowerPositionLimit,
            self.var_q,
            self.robot.upperPositionLimit)
        )
        self.opti.minimize(50 * self.translational_cost + self.rotation_cost + 0.02 * self.regularization_cost + 0.1 * self.smooth_cost)

        opts = {
            'ipopt':{
                'print_level':0,
                'max_iter':50,
                'tol':1e-6
            },
            'print_time':False,# print or not
            'calc_lam_p':False # https://github.com/casadi/casadi/wiki/FAQ:-Why-am-I-getting-%22NaN-detected%22in-my-optimization%3F
        }
        self.opti.solver("ipopt", opts)

        self.init_data = np.zeros(self.robot.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 6)


    def solve_ik(self, target_ee, current_l_arm_motor_q = None, current_l_arm_motor_dq = None):
        if current_l_arm_motor_q is not None:
            self.init_data = current_l_arm_motor_q
        self.opti.set_initial(self.var_q, self.init_data)



        self.opti.set_value(self.param_tf_l, target_ee.homogeneous)
        self.opti.set_value(self.var_q_last, self.init_data) # for smooth

        try:
            sol = self.opti.solve()
            # sol = self.opti.solve_limited()

            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_l_arm_motor_dq is not None:
                v = current_l_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.robot, self.robot_data, sol_q, v, np.zeros(self.robot.nv))



            return sol_q, sol_tauff
        
        except Exception as e:
            print(f"ERROR in convergence, plotting debug info.{e}")

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_l_arm_motor_dq is not None:
                v = current_l_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(self.robot, self.robot_data, sol_q, v, np.zeros(self.robot.nv))

            print(f"sol_q:{sol_q} \nmotorstate: \n{current_l_arm_motor_q} \nleft_pose: \n{target_ee}")


            # return sol_q, sol_tauff
            return current_l_arm_motor_q, np.zeros(self.robot.nv)


    # def step(self, head_rmat, left_pose, right_pose, left_qpos, right_qpos):
    def step(self, head_rmat, right_pose, trigger):
        # enter VR will cause the arm explode, setting to feasible position
        if(right_pose[0]> 1.3):
            right_pose[:3] = np.array([0.6, 0.0, 0.8])
        if self.print_freq:
            start = time.time()
        # root_states里包含了所有的asset的root位置
        if self.right_compute_flag:
            prev_pose = right_pose[0:7]
            prev_rotation = R.from_quat(right_pose[3:]).as_matrix()
            self.right_compute_flag = False
        else:
            prev_pose = self.right_root_states[0:7].numpy()
            prev_rotation = R.from_quat(prev_pose[3:]).as_matrix()

        prev_SE3 = pin.SE3(prev_rotation, prev_pose[:3])
        

        # self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float) # current left_hand 更新手部位置
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)

    
        target_pos = right_pose[:3]
        
        target_rotation = R.from_quat(right_pose[3:]).as_matrix()

        # rotation_z = R.from_euler('z', 90, degrees=True).as_matrix()

        # # 绕旋转后的 z 轴旋转 90 度
        rotation_x = R.from_euler('x', 90, degrees=True).as_matrix()
        
        # # 先绕 x 轴旋转
        target_rotation_transformed = target_rotation @ rotation_x

        target_pose = pin.SE3(target_rotation_transformed , target_pos)
        

        ### ----- 计算IK

        
        max_iter = 1000
        tol = 1e-4
        # 获取当前dof信息（包含v）
        
        
        dof_state = self.gym.get_actor_dof_states(self.env, self.cowa_handle, self.dof_cowa)
        q = dof_state['pos']

        

        ### -----gripper
        if trigger:
            griper_width = 0.02  # 最小宽度（闭合状态）
        else:
            griper_width = 0.11  # 夹爪打开
        # 归一化并转化为控制信号
        griper_cmd = (np.clip(griper_width, 0.02, 0.11) - 0.02) / 0.09 * 100
        #gripper
        self.cmd[0] = griper_cmd
        
        speed = np.zeros(self.robot.nv)

        q, speed = self.solve_ik(target_pose, q, speed)
        
        ####----publisher
        self.lower_limit = self.robot.lowerPositionLimit
        self.upper_limit = self.robot.upperPositionLimit
        q_clip = np.clip(q, self.lower_limit, self.upper_limit)
        self.cmd[1:] = q_clip
        msg={'action': list(self.cmd)}
        # print("success")
        # print(msg)
        # self.cmd_pub.send(topic='arm_action', msg=msg)      
        
        #感觉像是一步把所有的root全部定义掉（right，left，cowa）
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

        cowa_states = np.zeros(self.dof_cowa, dtype=gymapi.DofState.dtype)
        if self.first_compute_flag:
            cowa_states['pos'] = [0, -0.5, 0.8, 0, 0, 0]
            self.first_compute_flag = False
        else:
            cowa_states['pos'] = q

        # print(f"cowa_states: {cowa_states}")
        self.gym.set_actor_dof_states(self.env, self.cowa_handle, cowa_states, gymapi.STATE_POS)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        # self.gym.set_camera_location(self.left_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
        #                              gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        # self.gym.set_camera_location(self.right_camera_handle,
        #                              self.env,
        #                              gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
        #                              gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
        # left_image = self.gym.get_camera_image(self.sim, self.env, self.left_camera_handle, gymapi.IMAGE_COLOR)
        # right_image = self.gym.get_camera_image(self.sim, self.env, self.right_camera_handle, gymapi.IMAGE_COLOR)
        # left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        # right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        if self.print_freq:
            end = time.time()
            print('Frequency:', 1 / (end - start))

        # return left_image, right_image

    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

# def realsense_frame_process(self, shm_name, img_shape):
#     try:
#         existing_shm = shared_memory.SharedMemory(name=shm_name)
#         img_array = np.ndarray((img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=existing_shm.buf)
#         print('img________________',img_shape[0], img_shape[1])
#         realsense_tool = RealSenseTools() 
#         while True:
#             color_img, depth_img, depth_frame, depth_colormap = realsense_tool.recv() # 使用 self.realsense_tool.recv()
#             if color_img.shape == img_array.shape and color_img.dtype == img_array.dtype:
#                 #  print("copy img")
#                     np.copyto(img_array, color_img) # 使用 np.copyto 复制数据
#                 #  cv2.imshow(color_img)
#                 #  cv2.waitKey(1)
#             else:
#                 print("error,",color_img.shape,img_array.shape)
#     except Exception as e:
#         print(f"RealSense Capture Process Error: {e}")
#     finally:
#         print("RealSense capture process finished.")
#         existing_shm.close()

if __name__ == '__main__':
    teleoperator = VuerTeleop('inspire_hand.yml')
    simulator = Sim()
    realsense = RealSenseTools()
    # realsense = RealSenseTools()
    # realsense.get_intrinsics()
    # plt.ion()  
    # fig, ax = plt.subplots()
    # im = ax.imshow(np.random.rand(840, 480, 3)) 
    try:
        while True:
            head_rmat, right_pose, trigger = teleoperator.step()
            simulator.step(head_rmat, right_pose, trigger)
            color_img, depth_img, depth_frame, depth_colormap = realsense.recv()
            color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            # im.set_data(color_img)
            # plt.pause(0.001)
            # np.copyto(teleoperator.img_array, color_img)
            np.copyto(teleoperator.img_array, np.hstack((color_img, color_img)))

    except KeyboardInterrupt:
        simulator.end()
        exit(0)
