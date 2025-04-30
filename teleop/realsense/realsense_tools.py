import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RealSenseTools:
    def __init__(self, camera_serial="748512060307", w=640, h=480, fps=30, depth_max_distance=1.0):
        # Set param
        self.depth_max_distance = depth_max_distance # 大于此距离的像素在depth img中将被截断

        # Init RealSense
        self.pipeline = rs.pipeline()  # 定义流程pipeline

        config = rs.config()  # 定义配置config
        config.enable_device(camera_serial)
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)  # 配置depth流
        config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)  # 配置color流

        self.profile = self.pipeline.start(config) 

        # 获取深度传感器的深度比例
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        align_to = rs.stream.color  # 对齐到color流
        self.align = rs.align(align_to)
        for _ in range(10):
            self.recv()
    
    def get_intrinsics(self):
        # 获取深度流和彩色流的内参
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()

        depth_intrinsics = depth_profile.get_intrinsics()
        color_intrinsics = color_profile.get_intrinsics()

        print({'fx': color_intrinsics.fx, 'fy': color_intrinsics.fy,
            'ppx': color_intrinsics.ppx, 'ppy': color_intrinsics.ppy,
            'height': color_intrinsics.height, 'width': color_intrinsics.width})

        return color_intrinsics, depth_intrinsics
    
    def recv_depth(self):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        depth_image_z16: np.array = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit: np.array = cv2.convertScaleAbs(depth_image_z16, alpha=255 * self.depth_scale/self.depth_max_distance)  # 深度图（8位）

        color_image_array: np.array = np.asanyarray(color_frame.get_data())  # RGB图

        # 伪彩色图
        depth_colormap_array: np.array = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)

        return color_image_array, depth_image_8bit, aligned_depth_frame, depth_colormap_array
    
    def recv(self):
        frames = self.pipeline.wait_for_frames()  # 等待获取图像帧
        aligned_frames = self.align.process(frames)  # 获取对齐帧
        aligned_depth_frame = aligned_frames.get_depth_frame()  # 获取对齐帧中的depth帧
        color_frame = aligned_frames.get_color_frame()  # 获取对齐帧中的color帧

        depth_image_z16: np.array = np.asanyarray(aligned_depth_frame.get_data())  # 深度图（默认16位）
        depth_image_8bit: np.array = cv2.convertScaleAbs(depth_image_z16, alpha=255 * self.depth_scale/self.depth_max_distance)  # 深度图（8位）
        color_image_array: np.array = np.asanyarray(color_frame.get_data())  # RGB图

        # 伪彩色图
        depth_colormap_array: np.array = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
        # cv2.imshow("color and depth", color_image_array)
        # cv2.waitKey(1)
        return color_image_array, depth_image_8bit, aligned_depth_frame, depth_colormap_array

class RealSenseTools_T265:
    def __init__(self, camera_serial="908412110805"):
        self.pipelinie = rs.pipeline()
        config = rs.config()
        config.enable_device(camera_serial)
        config.enable_stream(rs.stream.pose, -1, rs.format.motion_xyz32f, 20)
        profile =  self.pipelinie.start(config)
        device = profile.get_device()
        self.sensors = device.query_sensors()
        self.pose_sensor = self.sensors[0]
    
    def recv(self):
        frames = self.pipelinie.wait_for_frames()
        pose_frame = frames.get_pose_frame()
        pose_data = pose_frame.get_pose_data()
        translation = pose_data.translation
        rotation = pose_data.rotation
        pose_array = np.array([translation.x, translation.y, translation.z,
                                rotation.x, rotation.y, rotation.z, rotation.w])
        # print(pose_array)
        return pose_array

if __name__ == "__main__":
    # realsense = RealSenseTools()
    # realsense.get_intrinsics()

    # while True:
    #     color_img, depth_img, depth_frame, depth_colormap = realsense.recv()
    #     # cv2.imshow("color and depth", np.concatenate([color_img, depth_colormap], axis=1))
    #     cv2.imshow("color and depth", color_img)
    #     cv2.imshow("depth", depth_img)
    #     cv2.waitKey(1)
    realsense = RealSenseTools_T265()
    # while True:
        # pos_array = realsense.recv()
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter([], [], [])

    x_coords = []
    y_coords = []
    z_coords = []

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Real-time T265 Position Visualization')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    plt.ion()  # 开启交互模式，允许实时更新
    while True:
        pos_array = realsense.recv()
        print(pos_array)
        translation = pos_array[:3]
        x, y, z = translation[0], translation[1], translation[2]
        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)
        ax.clear()
        ax.scatter(x_coords, y_coords, z_coords)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Real-time T265 Position Visualization')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()