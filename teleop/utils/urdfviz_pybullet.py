import pybullet as p
import pybullet_data
import time
import numpy as np
def visualize_urdf_with_link_axis(urdf_path, link_name):

    physicsClient = p.connect(p.GUI)
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.setTimeStep(1./240.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # p.loadURDF("plane.urdf")
    robotId = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=70, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])
    link_index = -1

    for i in range(p.getNumJoints(robotId)): 
        jointInfo = p.getJointInfo(robotId, i)
        linkName = jointInfo[12].decode('utf-8') 
        if linkName == link_name:
            link_index = i
            break

    if link_index != -1:
        # 创建坐标轴的视觉形状 (红色 X, 绿色 Y, 蓝色 Z)
        axis_length = 0.1  
        x_axis_visual = p.createVisualShape(p.GEOM_LINE, lineFrom=[0,0,0], lineTo=[axis_length,0,0], rgbaColor=[1,0,0,1], lineWidth=3)
        y_axis_visual = p.createVisualShape(p.GEOM_LINE, lineFrom=[0,0,0], lineTo=[0,axis_length,0], rgbaColor=[0,1,0,1], lineWidth=3)
        z_axis_visual = p.createVisualShape(p.GEOM_LINE, lineFrom=[0,0,0], lineTo=[0,0,axis_length], rgbaColor=[0,0,1,1], lineWidth=3)

        axis_body_id = p.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1, 
            baseVisualShapeIndex=x_axis_visual, 
            basePosition=[0, 0, 0],
            baseOrientation=[0, 0, 0, 1]
        )
        p.changeVisualShape(axis_body_id, -1, shapeIndex=y_axis_visual, flags=p.ADD_REPLACE_VISUAL_SHAPE) 
        p.changeVisualShape(axis_body_id, -1, shapeIndex=z_axis_visual, flags=p.ADD_REPLACE_VISUAL_SHAPE) 
    else:
        print(f"Link '{link_name}' not found in URDF.")
        axis_body_id = -1 
    
    


    while True:
        p.stepSimulation()

        if link_index != -1 and axis_body_id != -1:
            link_state = p.getLinkState(robotId, link_index)
            link_position = link_state[0]    
            link_orientation = link_state[1] # 四元数
            p.resetBasePositionAndOrientation(axis_body_id, link_position, link_orientation)


        time.sleep(1./240.)

    p.disconnect()


def draw_coordinate_frame(urdf_path, link_name):
    """ Draws coordinate frame (XYZ axes) at a given position with a given orientation (quaternion) in PyBullet. """

    physicsClient = p.connect(p.GUI)

    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.setTimeStep(1./240.)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    robotId = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=70, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    link_index = -1 
    for i in range(p.getNumJoints(robotId)): 
        jointInfo = p.getJointInfo(robotId, i)
        linkName = jointInfo[12].decode('utf-8')
        if linkName == link_name:
            link_index = i
            break
        
    axis_length = 1
    
    link_state = p.getLinkState(robotId, link_index)
    link_position = link_state[0]    
    link_orientation = link_state[1] 

    # Convert quaternion to rotation matrix
    rotation_matrix = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape(3, 3)

    x_axis = np.array([axis_length, 0, 0])
    y_axis = np.array([0, axis_length, 0])
    z_axis = np.array([0, 0, axis_length])

    x_axis_world = link_position + np.dot(rotation_matrix, x_axis)
    y_axis_world = link_position + np.dot(rotation_matrix, y_axis)
    z_axis_world = link_position + np.dot(rotation_matrix, z_axis)

    p.addUserDebugLine(link_position, x_axis_world, lineColorRGB=[1, 0, 0], lifeTime=0)  
    p.addUserDebugLine(link_position, y_axis_world, lineColorRGB=[0, 1, 0], lifeTime=0)  
    p.addUserDebugLine(link_position, z_axis_world, lineColorRGB=[0, 0, 1], lifeTime=0)  


    while True:
        p.removeAllUserDebugItems()
        p.stepSimulation()

        if link_index != -1:
            link_state = p.getLinkState(robotId, link_index)
            link_position = link_state[0]    
            link_orientation = link_state[1] 

            p.resetBasePositionAndOrientation(21, link_position, link_orientation)
            rotation_matrix = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape(3, 3)

            x_axis = np.array([axis_length, 0, 0])
            y_axis = np.array([0, axis_length, 0])
            z_axis = np.array([0, 0, axis_length])

            x_axis_world = link_position + np.dot(rotation_matrix, x_axis)
            y_axis_world = link_position + np.dot(rotation_matrix, y_axis)
            z_axis_world = link_position + np.dot(rotation_matrix, z_axis)

            p.addUserDebugLine(link_position, x_axis_world, lineColorRGB=[1, 0, 0], lifeTime=0)  
            p.addUserDebugLine(link_position, y_axis_world, lineColorRGB=[0, 1, 0], lifeTime=0)  
            p.addUserDebugLine(link_position, z_axis_world, lineColorRGB=[0, 0, 1], lifeTime=0)  

if __name__ == '__main__':
    urdf_file = "../assets/cowa_legged_wheel_arm/urdf/cowa_legged_wheel_arm.urdf"
    target_link_name = "hand_center"  
    # visualize_urdf_with_link_axis(urdf_file, target_link_name)
    draw_coordinate_frame(urdf_file, target_link_name)