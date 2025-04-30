import zarr
import argparse
import cv2
from utils.imagecodecs_numcodecs import register_codecs
from utils.msg_communicator import MsgPublisher
import numpy as np  # 导入 numpy
import os
register_codecs()

def replay(file_path, m):
    cmd_pub = MsgPublisher(port=22222)
    dataset = zarr.open(file_path, mode='r+')  # 修改为读写模式
    if '/data/memory' not in dataset:
        print(f"'/data/memory' not found in {file_path}. Creating it.")
        memory_len = len(dataset['/data/ee_pos'])
        dataset.create_dataset('/data/memory', shape=(memory_len,), dtype=np.uint8, fillvalue=0)
        print(f"Created '/data/memory' with length {memory_len} and initialized with 0.")
    else:
        memory_len = len(dataset['/data/memory'])
    i = 0
    while i < len(dataset['/data/ee_pos']): # 使用 while 循环更方便控制 i 的增长
        memory = dataset['/data/memory'][i]
        img = dataset['/data/camera_img'][i]
        # print(dataset['/data/gripper_pos'][i])
        # print(f"Original Memory at index {i}: {memory}")
        info_text = f"Index: {i}, Memory: {memory}, M_Count: {m}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('replay', img)
        key = cv2.waitKey(0)
        if key == ord('o'):
            break
        if key == ord('k'):
            i = i + 5
        elif key == ord('j'):
            i = i - 2
        elif key == ord('m'):  # 按下 'm' 键修改 memory
            print(f"Marking memory for indices {i} to {min(i + 10, memory_len) - 1} with value 1")
            for j in range(10):
                index_to_modify = i + j
                if index_to_modify < memory_len:
                    dataset['/data/memory'][index_to_modify] = 1
        elif key == ord('n'):  # 按下'n' 键修改 memory
            print(f"Marking memory for indices {i} to {min(i + 10, memory_len) - 1} with value 0")
            for j in range(30):
                index_to_modify = i + j
                if index_to_modify < memory_len:
                    dataset['/data/memory'][index_to_modify] = 0
        i += 1 # 每次循环结束后 i 增加 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', '-f', type=str,
                        help='Path to the directory containing the record files')
    args = parser.parse_args()

    record_dir = args.record_file
    m = 1
    try:
        file_names = os.listdir(record_dir)
        for file_name in file_names:
            full_file_path = os.path.join(record_dir, file_name)
            print('editing:', full_file_path)
            replay(full_file_path, m)
            m = m + 1
    except OSError as e:
        print(f"Error reading directory '{record_dir}': {e}")
