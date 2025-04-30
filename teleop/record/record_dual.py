import io
import time
import datetime
import base64
import zarr
import gc
import numpy as np
import cv2

from threading import Thread
from pathlib import Path
from multiprocessing import Process, Queue, Event, shared_memory
from replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs, JpegXl
from realsense.realsense_tools import RealSenseTools
from utils.msg_communicator import MsgPublisher, MsgSubscriber
register_codecs()

# camera_serial
# 218622275838 d405
# 748512060307 d415 
class ArmMsg:
    def __init__(self):
        self.joints_pos = np.zeros(7, dtype=np.float32)  # 假设关节数为7,如joints_vel果不是请调整
        self.joints_vel = np.zeros(7, dtype=np.float32)
        self.camera_img = None
        self.arm_state_sub = MsgSubscriber(ip='192.168.1.3', port=22222, topic='arm_state')
        # self.camera_data_sub = MsgSubscriber(ip='192.168.1.port=22223, topic='camera_data')
        self.camera_data_sub_d405 = RealSenseTools(cameral_serial="218622275838") 
        self.camera_data_sub_d415 = RealSenseTools(cameral_serial="748512060307")

    def run_arm_state_sub(self, terminal):
        while not terminal.is_set():
            arm_state = self.arm_state_sub.recv()
            self.joints_pos = np.array(arm_state['joints_pos'], dtype=np.float32)
            self.joints_vel = np.array(arm_state['joints_vel'], dtype=np.float32)
        print('Stop run_arm_state_sub')
        return

    def run_camera_data_sub(self, terminal, Q: Queue):
        while not terminal.is_set():
            color_img1, depth_img1, depth_frame1, depth_colormap1 = self.camera_data_sub_d415.recv_depth()
            color_img2, depth_img2, depth_frame2, depth_colormao2 = self.camera_data_sub_d405.recv_depth()
            Q.put({
                'joints_pos': self.joints_pos.copy(), 
                'joints_vel': self.joints_vel.copy(), 
                'camera_img1': color_img1, 
                "depth_img1": depth_img1,
                'camera_img2': color_img2, 
                "depth_img2": depth_img2
            })
        print('Stop run_camera_data_sub')
        return

def arm_state_recv(terminate, recording, Q: Queue):
    arm_msg = ArmMsg()
    th1 = Thread(target=arm_msg.run_arm_state_sub, args=(terminate,))
    th2 = Thread(target=arm_msg.run_camera_data_sub, args=(terminate, Q))
    th1.start()
    th2.start()
    th1.join()
    th2.join()

def set_shared_memory(data: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    return shm.name, data.shape, data.dtype

def record(terminate, recording, Q: Queue, save_Q: Queue):
    f_press_time = 0
    t_press_time = 0
    cv2.namedWindow('img1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    joints_pos_lst = []
    joints_vel_lst = []
    camera_img_lst1 = []
    depth_img_lst1 = []
    camera_img_lst2 = []
    depth_img_lst2 = []
    memory_lst = []
    saved_cnt = 0
    while not terminate.is_set():
        data = Q.get()
        camera_img = data['camera_img1'].copy()
        camera_img1 = data['camera_img2'].copy()
        key = cv2.waitKey(1) & 0xFF
        m_pressed = (key == ord('m'))
        t = time.time()
        if t - f_press_time < 3:
            cv2.putText(camera_img, 'Saved', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if t - t_press_time < 3:
            cv2.putText(camera_img, 'Truncted', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if recording.is_set():
            joints_pos_lst.append(data['joints_pos'])
            joints_vel_lst.append(data['joints_vel'])
            camera_img_lst1.append(data['camera_img1'])
            depth_img_lst1.append(data['depth_img1'])
            camera_img_lst2.append(data['camera_img2'])
            depth_img_lst2.append(data['depth_img2'])
            # 检测m键的状态并存储
            memory_lst.append(1 if m_pressed else 0)
            if m_pressed:
                cv2.putText(camera_img, 'Memory Key Pressed', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(camera_img, 'Recording', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(camera_img, f'Saved {saved_cnt}', (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow('img1', camera_img)
        cv2.imshow('img2', camera_img1)
        if key == ord('q'):
            terminate.set()
        elif key == ord('r'):
            recording.set()
        elif key == ord('f'):
            if recording.is_set():
                f_press_time = time.time()
                recording.clear()
                # compose data
                joints_pos = np.stack(joints_pos_lst)
                joints_vel = np.stack(joints_vel_lst)
                camera_imgs1 = np.stack(camera_img_lst1)
                depth_imgs1 = np.stack(depth_img_lst1)
                camera_imgs2 = np.stack(camera_img_lst2)
                depth_imgs2 = np.stack(depth_img_lst2)
                memory_array = np.array(memory_lst, dtype=np.int32)
                save_Q.put((set_shared_memory(joints_pos), set_shared_memory(joints_vel), 
                            set_shared_memory(camera_imgs1), set_shared_memory(depth_imgs1),
                            set_shared_memory(camera_imgs2), set_shared_memory(depth_imgs2),
                            set_shared_memory(memory_array),))
                saved_cnt += 1
                joints_pos_lst.clear()
                joints_vel_lst.clear()
                camera_img_lst1.clear()
                depth_img_lst1.clear()
                camera_img_lst2.clear()
                depth_img_lst2.clear()
                memory_lst.clear()
                gc.collect()
        elif key == ord('t'):
            if recording.is_set():
                t_press_time = time.time()
                joints_pos_lst.clear()
                joints_vel_lst.clear()
                camera_img_lst1.clear()
                depth_img_lst1.clear()
                camera_img_lst2.clear()
                depth_img_lst2.clear()
                recording.clear()
                gc.collect()
    cv2.destroyAllWindows()
    print('Stop record')
    return

def get_shared_memory(shm_name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    print(shared_array.shape)   
    return existing_shm, shared_array

def save_episode(terminate, save_Q: Queue):
    dataset_dir = Path(__file__).parent.parent/'records'
    dataset_dir.mkdir(exist_ok=True, parents=True)
    img_compressor = JpegXl(level=99, numthreads=8)
    while not terminate.is_set():
        joints_pos_meta, joints_vel_meta, camera_imgs_meta1, depth_imgs_meta1, camera_imgs_meta2, depth_imgs_meta2, memory_meta = save_Q.get()
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        dataset_path = dataset_dir/f"{timestamp_str}.zarr"
        print('Saving ...')
        with zarr.DirectoryStore(str(dataset_path)) as store:
            root = zarr.group(store=store)
            joints_pos_shm, joints_pos = get_shared_memory(*joints_pos_meta)
            joints_vel_shm, joints_vel = get_shared_memory(*joints_vel_meta)
            camera_imgs_shm1, camera_imgs1 = get_shared_memory(*camera_imgs_meta1)
            depth_imgs_shm1, depth_imgs1 = get_shared_memory(*depth_imgs_meta1)
            camera_imgs_shm2, camera_imgs2 = get_shared_memory(*camera_imgs_meta2)
            depth_imgs_shm2, depth_imgs2 = get_shared_memory(*depth_imgs_meta2)
            memory_shm, memory_array = get_shared_memory(*memory_meta)

            root.create_dataset('data/joints_pos', data=joints_pos, chunks=joints_pos.shape)
            root.create_dataset('data/joints_vel', data=joints_vel, chunks=joints_vel.shape)
            # chunks与compressor可能需要修改
            root.create_dataset('data/camera_img1', data=camera_imgs1, chunks=camera_imgs1[:1].shape, compressor=img_compressor)
            root.create_dataset('data/depth_img1', data=depth_imgs1, chunks=depth_imgs1[:1].shape, compressor=img_compressor)
            root.create_dataset('data/camera_img2', data=camera_imgs2, chunks=camera_imgs2[:1].shape, compressor=img_compressor)
            root.create_dataset('data/depth_img2', data=depth_imgs2, chunks=depth_imgs2[:1].shape, compressor=img_compressor)
            root.create_dataset('data/memory', data=memory_array, chunks=memory_array.shape)
            
            joints_pos_shm.close()
            joints_pos_shm.unlink()
            joints_vel_shm.close()
            joints_vel_shm.unlink()
            camera_imgs_shm1.close()
            camera_imgs_shm1.unlink()
            depth_imgs_shm1.close()
            depth_imgs_shm1.unlink()
            camera_imgs_shm2.close()
            camera_imgs_shm2.unlink()
            depth_imgs_shm2.close()
            depth_imgs_shm2.unlink()
            memory_shm.close()
            memory_shm.unlink()
        print(f'Successfully save to: {str(dataset_path)}')

def main():
    Q = Queue(maxsize=10000)
    save_Q = Queue(maxsize=1)
    terminate = Event()
    recording = Event()
    p1 = Process(target=arm_state_recv, args=(terminate, recording, Q))
    p2 = Process(target=save_episode, args=(terminate, save_Q))
    p1.start()
    p2.start()
    record(terminate, recording, Q, save_Q)
    p1.join()
    p2.join()

def test():
    pass

if __name__ == '__main__':
    main()
    # test()
