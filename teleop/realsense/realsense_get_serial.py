import pyrealsense2 as rs

def find_camera_serials():
    """Finds and returns a list of serial numbers of connected RealSense cameras."""
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        serials.append(dev.get_info(rs.camera_info.serial_number))
    return serials

if __name__ == "__main__":
    serials = find_camera_serials()
    if serials:
        print("Available RealSense camera serial numbers:")
        for serial in serials:
            print(serial)
    else:
        print("No RealSense cameras found.")


# 218622275838
# 748512060307