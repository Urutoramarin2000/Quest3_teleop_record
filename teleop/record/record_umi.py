import io
import time
import datetime
import base64
import zarr
import gc
import numpy as np
from threading import Thread
from pathlib import Path
from multiprocessing import Process, Queue, Event, shared_memory
# from .replay_buffer import ReplayBuffer
from utils.msg_communicator import MsgSubscriber
from utils.imagecodecs_numcodecs import register_codecs, JpegXl
from realsense.realsense_tools import RealSenseTools, RealSenseTools_T265

register_codecs()

class ArmMsg:
    def __init__(self):
        self.camera_img = None
        self.gripper_pos = np.zeros(1, dtype=np.float32) 
        self.ee_pos = np.zeros(7, dtype=np.float32)
        
        self.camera_data_sub = MsgSubscriber(ip='192.168.1.3',port=22223, topic='camera_data')
        self.gripper_state_sub = MsgSubscriber(ip='192.168.1.3', port=22222, topic='arm_state')
        self.pos_data_sub = RealSenseTools_T265()

    def run_eepos_state_sub(self, terminal):
        while not terminal.is_set():
            eepos_state = self.pos_data_sub.recv()
            self.ee_pos = eepos_state
        print('Stop run_arm_state_sub')
        return
    
    def run_gripper_state_sub(self, terminal):
        while not terminal.is_set():
            gripper_state = self.gripper_state_sub.recv()
            self.gripper_pos = gripper_state['joints_pos'][0]
            # print(self.gripper_pos)
        print('Stop run_arm_state_sub')
        return

    def run_camera_data_sub(self, terminal, Q: Queue):
        import av
        rawData = io.BytesIO()
        cur_pos = 0
        can_decode_h264 = False
        while not terminal.is_set():
            msg = self.camera_data_sub.recv()
            # Decode h264
            t1 = time.perf_counter()
            camera_bytes = base64.b64decode(msg['camera_data'].encode('utf-8')) # encode() 将 str 转换为 bytes
            rawData.write(camera_bytes)
            rawData.seek(cur_pos)
            if cur_pos == 0:
                container = av.open(rawData, format='h264', mode='r')
                original_codec_ctx = container.streams.video[0].codec_context
                codec = av.codec.CodecContext.create(original_codec_ctx.name, 'r')
            cur_pos += len(camera_bytes)
            for packet in container.demux():
                if packet.size == 0:
                    continue
                can_decode_h264 = True if packet.is_keyframe else can_decode_h264
                # print('can decode', can_decode_h264)
                if can_decode_h264:
                    frames = codec.decode(packet)
                    for frame in frames:
                        self.camera_img = frame.to_ndarray(format='bgr24')
                        Q.put({'ee_pos':self.ee_pos[:3],'ee_rot': self.ee_pos[3:],'gripper_pos': self.gripper_pos,'camera_img': self.camera_img})
        print('Stop run_camera_data_sub')
        return


def arm_state_recv(terminate, recording, Q: Queue):
    arm_msg = ArmMsg()
    th1 = Thread(target=arm_msg.run_eepos_state_sub, args=(terminate,))
    th2 = Thread(target=arm_msg.run_camera_data_sub, args=(terminate, Q))
    th3 = Thread(target=arm_msg.run_gripper_state_sub, args=(terminate,))

    th1.start()
    th2.start()
    th3.start()

    th1.join()
    th2.join()
    th3.join()


def set_shared_memory(data: np.ndarray):
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shared_array = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shared_array[:] = data[:]
    return shm.name, data.shape, data.dtype.str  

def get_shared_memory(shm_name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    shared_array = np.ndarray(shape, dtype=np.dtype(dtype), buffer=existing_shm.buf)
    return existing_shm, shared_array

def record(terminate, recording, Q: Queue, save_Q: Queue):
    import cv2
    f_press_time = 0
    t_press_time = 0
    ee_pos_lst = []
    ee_rot_lst = []
    gripper_pos_lst =[]
    camera_img_lst = []
    # depth_img_lst = []
    memory_lst = []
    saved_cnt = 0
    while not terminate.is_set():
        data = Q.get()
        camera_img = data['camera_img'].copy()
        # print(data['ee_pos'].copy())
        # depth_img = data['depth_img'].copy()
        t = time.time()
        key = cv2.waitKey(1) & 0xFF
        m_pressed = (key == ord('m'))
        if t - f_press_time < 3:
            cv2.putText(camera_img, 'Saved', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if t - t_press_time < 3:
            cv2.putText(camera_img, 'Truncted', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        if recording.is_set():
            ee_pos_lst.append(data['ee_pos'])
            ee_rot_lst.append(data['ee_rot'])
            gripper_pos_lst.append(data['gripper_pos'])
            camera_img_lst.append(data['camera_img'])
            # depth_img_lst.append(data['depth_img'])
            memory_lst.append(1 if m_pressed else 0)
            if m_pressed:
                cv2.putText(camera_img, 'Memoring', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(camera_img, 'Recording', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        cv2.putText(camera_img, f'Saved {saved_cnt}', (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow('img', camera_img)
        if key == ord('q'):
            terminate.set()
        elif key == ord('r'):
            recording.set()
        elif key == ord('f'):
            if recording.is_set():
                print('saving ...')
                f_press_time = time.time()
                recording.clear()
                # compose data
                ee_pos = np.stack(ee_pos_lst)
                ee_rot = np.stack(ee_rot_lst)
                gripper_pos = np.stack(gripper_pos_lst)
                # for i, img in enumerate(camera_img_lst):
                #     print(f"Image {i}: shape={img.shape}, dtype={img.dtype}")
                camera_imgs = np.stack(camera_img_lst)
                # print(camera_imgs.dtype, camera_imgs.shape)
                # depth_imgs = np.stack(depth_img_lst)
                memory_array = np.array(memory_lst, dtype=np.int32)
                save_Q.put((set_shared_memory(ee_pos), 
                            set_shared_memory(ee_rot),
                            set_shared_memory(camera_imgs), 
                            set_shared_memory(memory_array),
                            set_shared_memory(gripper_pos),
                           ))
                saved_cnt += 1
                ee_pos_lst.clear()
                ee_rot_lst.clear()
                gripper_pos_lst.clear()
                camera_img_lst.clear()
                # depth_img_lst.clear()
                memory_lst.clear()
                gc.collect()
        elif key == ord('t'):
            if recording.is_set():
                t_press_time = time.time()
                ee_pos_lst.clear()
                ee_rot_lst.clear()
                gripper_pos_lst.clear()
                camera_img_lst.clear()
                # depth_img_lst.clear()
                memory_lst.clear()
                recording.clear()
                gc.collect()
    cv2.destroyAllWindows()
    print('Stop record')
    return



def save_episode(terminate, save_Q: Queue):
    dataset_dir = Path(__file__).parent.parent/'records'
    dataset_dir.mkdir(exist_ok=True, parents=True)
    img_compressor = JpegXl(level=99, numthreads=8)
    while not terminate.is_set():
        ee_pos_meta, ee_rot_meta, camera_imgs_meta, memory_meta, gripper_pos_meta = save_Q.get()
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        dataset_path = dataset_dir/f"{timestamp_str}.zarr"
        print('Saving ...')
        with zarr.DirectoryStore(str(dataset_path)) as store:
            root = zarr.group(store=store)
            ee_pos_shm, ee_pos = get_shared_memory(*ee_pos_meta)
            ee_rot_shm, ee_rot = get_shared_memory(*ee_rot_meta)
            camera_imgs_shm, camera_imgs = get_shared_memory(*camera_imgs_meta)
            # depth_imgs_shm, depth_imgs = get_shared_memory(*depth_imgs_meta)
            memory_shm, memory_array = get_shared_memory(*memory_meta)
            gripper_shm, gripper_pos = get_shared_memory(*gripper_pos_meta)
            camera_imgs = np.array(camera_imgs, copy=True)
            print(camera_imgs.dtype)
            root.create_dataset('data/ee_pos', data=ee_pos, chunks=ee_pos.shape)
            root.create_dataset('data/ee_rot', data=ee_rot, chunks=ee_rot.shape)
            root.create_dataset('data/gripper_pos', data=gripper_pos, chunks=gripper_pos.shape)
            root.create_dataset('data/camera_img', data=camera_imgs, chunks=camera_imgs[:1].shape, compressor=img_compressor)
            # root.create_dataset('data/depth_img', data=depth_imgs, chunks=depth_imgs[:1].shape, compressor=img_compressor)
            root.create_dataset('data/memory', data=memory_array, chunks=memory_array.shape)
            
            ee_pos_shm.close()
            ee_pos_shm.unlink()
            ee_rot_shm.close()
            ee_rot_shm.unlink()
            gripper_shm.close()
            gripper_shm.unlink()
            camera_imgs_shm.close()
            camera_imgs_shm.unlink()
            # depth_imgs_shm.close()
            # depth_imgs_shm.unlink()
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
