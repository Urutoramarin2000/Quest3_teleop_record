import pybullet as p
import pybullet_data
import numpy as np
import time


class robot_solver:
    def __init__(self, robot_path, render=True):
        self.render = render
        if self.render:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setGravity(0, 0, -9.81)  # 设置重力

        # self.ROBOT_ID = p.loadURDF('../assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf', [0.0, 0, 0.0], useFixedBase=True)
        self.ROBOT_ID = p.loadURDF(robot_path, [0.0, 0, 0.0], useFixedBase=True)
        # self.ROBOT_ID = p.loadURDF('/home/cowa/Documents/TeleVision/assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf', [0.0, 0, 0.0], useFixedBase=True)
        self.EE_LINK_INDEX = 21  # 作为EE的link的索引, 19对应l_ace #20 对于r_ace

        self.num_joints = p.getNumJoints(self.ROBOT_ID)
        self.joint_info = [(p.getJointInfo(self.ROBOT_ID, i), i) for i in range(self.num_joints)]
        self.movable_joints = [info[1] for info in self.joint_info if p.getJointInfo(self.ROBOT_ID, info[1])[2] != p.JOINT_FIXED]

        self.movable_joint = [0., -0.8, 0.8, 0., 0., 0.]

        for i, movable_joint in enumerate(self.movable_joints):
            p.resetJointState(self.ROBOT_ID, movable_joint, self.movable_joint[i])


    def draw_coordinate_frame(self, position, orientation, axis_length=1):
        """ Draws coordinate frame (XYZ axes) at a given position with a given orientation (quaternion) in PyBullet. """
    
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

        x_axis = np.array([axis_length, 0, 0])
        y_axis = np.array([0, axis_length, 0])
        z_axis = np.array([0, 0, axis_length])

        x_axis_world = position + np.dot(rotation_matrix, x_axis)
        y_axis_world = position + np.dot(rotation_matrix, y_axis)
        z_axis_world = position + np.dot(rotation_matrix, z_axis)

        p.addUserDebugLine(position, x_axis_world, lineColorRGB=[1, 0, 0], lifeTime=0)  # X-axis (Red)
        p.addUserDebugLine(position, y_axis_world, lineColorRGB=[0, 1, 0], lifeTime=0)  # Y-axis (Green)
        p.addUserDebugLine(position, z_axis_world, lineColorRGB=[0, 0, 1], lifeTime=0)  # Z-axis (Blue)


    def solve_fk(self, joint_rads):
        '''
        Input joint_rads
        Output xyz and rotation angle
        '''
        joint_positions = joint_rads
        for movable_joint_id, joint_position in zip(self.movable_joints, joint_positions):
            p.resetJointState(self.ROBOT_ID, movable_joint_id, joint_position)
        # 获取机械臂ee link状态
        link_state = p.getLinkState(self.ROBOT_ID, self.EE_LINK_INDEX)
        xyz_position = link_state[0]  # ee link的世界坐标系中的位置
        orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
        angle_axis = p.getAxisAngleFromQuaternion(orientation) # 转轴角
        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_positions[idx])
            p.stepSimulation()
            # for _ in range(500):
            #     p.stepSimulation()
            #     time.sleep(0.1)
        self.draw_coordinate_frame(xyz_position, orientation)
        ee_state = np.hstack((xyz_position, np.array(angle_axis[0]) * angle_axis[1]))
        return ee_state


    def quaternion_to_rotation_matrix(self,quaternion):
        '''
        Input quaternion [x y z w]
        Output rotation matrix R
        '''
        # 四元数到旋转矩阵的转换
        q = quaternion
        R = np.array([[1 - 2 * (q[2]**2 + q[3]**2), 2 * (q[1] * q[2] - q[3] * q[0]), 2 * (q[1] * q[3] + q[2] * q[0])],
                    [2 * (q[1] * q[2] + q[3] * q[0]), 1 - 2 * (q[1]**2 + q[3]**2), 2 * (q[2] * q[3] - q[1] * q[0])],
                    [2 * (q[1] * q[3] - q[2] * q[0]), 2 * (q[2] * q[3] + q[1] * q[0]), 1 - 2 * (q[1]**2 + q[2]**2)]])
        return R

    def create_homogeneous_matrix(self, position, quaternion):
        # 创建4x4的齐次变换矩阵
        R = self.quaternion_to_rotation_matrix(quaternion)  # 旋转矩阵
        T = np.eye(4)  # 创建4x4单位矩阵
        T[:3, :3] = R  # 将旋转矩阵放入T的上左部分
        T[:3, 3] = position  # 将平移部分放入T的右上部分
        return T

    def solve_fk_quat(self, joint_rads):
        '''
        Input joint_rads
        return the target_to_base_homogenious_matrix
        '''
        joint_positions = joint_rads
        for movable_joint_id, joint_position in zip(self.movable_joints, joint_positions):
            p.resetJointState(self.ROBOT_ID, movable_joint_id, joint_position)
        # 获取机械臂ee link状态
        link_state = p.getLinkState(self.ROBOT_ID, self.EE_LINK_INDEX)
        xyz_position = link_state[0]  # ee link的世界坐标系中的位置
        orientation = link_state[1]  # ee link的世界坐标系中的四元数姿态
        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_positions[idx])
            p.stepSimulation()
            # for _ in range(500):
            #     p.stepSimulation()
            #     time.sleep(0.1)
        self.draw_coordinate_frame(xyz_position, orientation)
        T = self.create_homogeneous_matrix(xyz_position, orientation)
        return T 

    # def solve_ik(self, target_pos_xyz: np.ndarray=[0.6, 0.0, 0.43], target_axis_of_rot: np.ndarray=[0, 0, 0]):
    def solve_ik(self, target_pos_xyz: np.ndarray=[0.98324077, 0.09237641, 0.55775578], target_axis_of_rot: np.ndarray=[-0.02569118192651871, 0.11636336587600085, 0.026930236058372834]):
        axis = np.array(target_axis_of_rot[:3])  # 提取旋转轴
        angle = np.linalg.norm(axis)  # 旋转角度（轴的模）
        if angle > 1e-6:  # 避免除以零
            axis_normalized = axis / angle  # 单位化旋转轴
        else:
            axis_normalized = [1, 0, 0]  # 默认值（无旋转时，旋转轴可选任意方向）
            angle = 0

        target_orientation = p.getQuaternionFromAxisAngle(axis_normalized, angle)

        # 调用 inverse kinematics 函数
        joint_angles = p.calculateInverseKinematics(self.ROBOT_ID, self.EE_LINK_INDEX, target_pos_xyz, target_orientation, restPoses=self.movable_joint)

        self.draw_coordinate_frame(target_pos_xyz, target_orientation)

        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_angles[idx])
            p.stepSimulation()

        return joint_angles
    # maybe error
    def solve_ik_quat(self, target_pos_xyz: np.ndarray = np.array([0.98324077, 0.09237641, 0.55775578]), target_quat_orn: np.ndarray = np.array([0, 0, 0, 1])):
        target_orientation = target_quat_orn # 直接使用输入的四元数
        # print('target_pos_xyz',target_pos_xyz, 'target_quat_orn',target_quat_orn)
        # 调用 inverse kinematics 函数
        joint_angles = p.calculateInverseKinematics(self.ROBOT_ID, self.EE_LINK_INDEX, target_pos_xyz, target_orientation, restPoses=self.movable_joint)

        self.draw_coordinate_frame(target_pos_xyz, target_orientation)

        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_angles[idx])
            p.stepSimulation()

        return joint_angles

    # maybe error
    def rotation_matrix_to_quaternion(self,R):
        '''        return joint_angles

        Input Rotation matrix
        Output Quaternion
        '''
        w = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
        x = (R[2, 1] - R[1, 2]) / (4.0 * w)
        y = (R[0, 2] - R[2, 0]) / (4.0 * w)
        z = (R[1, 0] - R[0, 1]) / (4.0 * w)
        
        return [w, x, y, z]
    
    def solve_ik_matrix(self, target_transform: np.ndarray=np.array([[1, 0, 0, 0.98324077], 
                                                           [0, 1, 0, 0.09237641], 
                                                           [0, 0, 1, 0.55775578], 
                                                           [0, 0, 0, 1]])):
        target_pos_xyz = target_transform[:3, 3]
        target_rot_matrix = target_transform[:3, :3]

        target_orientation = self.rotation_matrix_to_quaternion(target_rot_matrix)

        # 调用 inverse kinematics 函数 Quaternion in [x y z w]
        joint_angles = p.calculateInverseKinematics(self.ROBOT_ID, self.EE_LINK_INDEX, target_pos_xyz, target_orientation, restPoses=self.movable_joint)
        self.draw_coordinate_frame(target_pos_xyz, target_orientation)

        if self.render:
            for idx, movable_joint in enumerate(self.movable_joints):
                p.setJointMotorControl2(self.ROBOT_ID, movable_joint, p.POSITION_CONTROL, targetPosition=joint_angles[idx])
            p.stepSimulation()

        return joint_angles
    
    def close(self):
        p.disconnect()

if __name__ == "__main__":
    ik = robot_solver(render=True)
    t = time.time()
    # ik.solve_fk(joint_rads=[0.]*6)
    print(ik.solve_ik())
    print(time.time() - t)
