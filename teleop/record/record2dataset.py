import cv2
import zarr
import argparse
import datetime
import numpy as np
from pathlib import Path
from replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs, JpegXl

register_codecs()

def record2dataset(args):
    record_dir = Path(args.record_dir)
    now = datetime.datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    dataset_path = record_dir.parent/'dataset'/f"{timestamp_str}.zarr"
    dataset_path.parent.mkdir(exist_ok=True, parents=True)
    dataset = ReplayBuffer.create_empty_zarr(zarr.DirectoryStore(str(dataset_path)))
    img_compressor = JpegXl(level=99, numthreads=8)
    for zarr_path in record_dir.glob('*.zarr'):
        print(f'processing: {str(zarr_path)}')
        
        record = zarr.open(str(zarr_path), mode='r')
        joints_pos = np.array(record['data/joints_pos'])
        joints_pos[np.where(joints_pos[:, 0] < 88), 0] = 0
        joints_pos[np.where(joints_pos[:, 0] >= 88), 0] = 1
        joints_vel = record['data/joints_vel']
        camera_img = record['data/camera_img']
        memory = record['data/memory']
        resized_imgs =  np.stack([cv2.resize(img, (224,224)) for img in camera_img])
        episode_data = {'joints_pos': joints_pos, 'joints_vel': joints_vel, 'camera_img': resized_imgs, 'memory': memory}
        chunks = {'camera_img': resized_imgs[:1].shape,}
        compressors = {'camera_img': img_compressor,}
        dataset.add_episode(episode_data, chunks=chunks, compressors=compressors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_dir', '-d', type=str,
                        help='Path to the record dir')
    args = parser.parse_args()

    record2dataset(args)