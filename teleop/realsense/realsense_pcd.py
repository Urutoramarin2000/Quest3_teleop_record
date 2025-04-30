import pyrealsense2 as rs
import numpy as np
import cv2

try:
    # 创建 Pipeline 对象
    pipeline = rs.pipeline()

    # 配置数据流：深度流和彩色流
    config = rs.config()
    config.enable_device("218622275838")
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # 启动 Pipeline
    pipeline.start(config)

    # 创建点云计算对象和纹理映射对象
    pc = rs.pointcloud()
    align_to = rs.align(rs.stream.color) # 创建对齐对象
    

    while True:
        # 获取帧数据
        frames = pipeline.wait_for_frames()

        # 对齐帧
        aligned_frames = align_to.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame: # 确保深度帧和彩色帧都成功获取
            continue

        # 生成点云
        points = pc.calculate(depth_frame)
        pc.map_to(color_frame) # 纹理映射

        # 获取顶点和彩色数据
        vtx = np.asanyarray(points.get_vertices())
        pixels = np.asanyarray(color_frame.get_data())

        # 可视化彩色点云
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(vtx[:, 0], vtx[:, 1], vtx[:, 2], c=pixels.reshape(-1, 3)/255.0, s=0.1) # 使用彩色数据
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('RealSense Color Point Cloud')
        plt.show()


        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()