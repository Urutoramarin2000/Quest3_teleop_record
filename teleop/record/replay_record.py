import zarr
import argparse
import cv2
from utils.imagecodecs_numcodecs import register_codecs
from utils.msg_communicator import MsgPublisher

register_codecs()

def replay(args):
    cmd_pub = MsgPublisher(port=22222)
    dataset = zarr.open(args.record_file, mode='r')
    for i in range(len(dataset['/data/joints_pos'])):
        joints_pos = dataset['/data/joints_pos'][i]
        joints_vel = dataset['/data/joints_vel'][i]
        memory = dataset['/data/memory'][i]
        img = dataset['/data/camera_img'][i]
        # img = dataset['/data/depth_img'][i]
        # print(joints_pos, joints_vel)
        print(joints_pos[0])
        # print(memory)
        joints_pos[0] = 100 if joints_pos[0] > 85 else 0
        cmd_pub.send(topic='arm_action', msg={'action': joints_pos.tolist()})
        cv2.imshow('replay', img)
        if cv2.waitKey(0) == ord('q'):
            return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_file', '-f', type=str,
                        help='Path to the record file')
    args = parser.parse_args()

    replay(args)