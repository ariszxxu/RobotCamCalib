"""Publish the latest right-wrist D435 color frame into shared memory.

Read-only ROS subscriber. Does not open the camera device or send actuator
commands. System Python 3.8 owns rospy; the OpenCV window runs separately.
"""
import json
import os
import struct
import time

import rospy
from sensor_msgs.msg import CameraInfo, Image

SHM = "/dev/shm/right-d435-color.bin"
INFO = "/dev/shm/right-d435-color-info.json"
MAGIC = b"RSD435\0\0"


def bgr_array(msg):
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError("Unsupported color encoding " + msg.encoding)
    import numpy as np

    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    image = data[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def main():
    rospy.init_node("right_d435_cv2_source", anonymous=True, disable_signals=True)
    state = {"info": None, "seq": 0}

    def on_info(msg):
        state["info"] = {
            "width": msg.width,
            "height": msg.height,
            "stamp": msg.header.stamp.to_sec(),
            "frame_id": msg.header.frame_id,
            "distortion_model": msg.distortion_model,
            "D": list(msg.D),
            "K": list(msg.K),
        }

    def on_image(msg):
        image = bgr_array(msg)
        state["seq"] += 1
        header = struct.pack(
            "<8sQdIII",
            MAGIC,
            state["seq"],
            msg.header.stamp.to_sec(),
            image.shape[0],
            image.shape[1],
            3,
        )
        blob = header + image.tobytes()
        tmp = "%s.%d.tmp" % (SHM, os.getpid())
        with open(tmp, "wb") as handle:
            handle.write(blob)
        os.replace(tmp, SHM)

    rospy.Subscriber("/camera_r/color/camera_info", CameraInfo, on_info, queue_size=1)
    rospy.Subscriber(
        "/camera_r/color/image_raw",
        Image,
        on_image,
        queue_size=1,
        buff_size=8000000,
    )
    deadline = time.monotonic() + 8
    while state["info"] is None and time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(0.05)
    if state["info"] is None:
        raise RuntimeError("Right wrist camera_info was not received")
    with open(INFO, "w") as handle:
        json.dump(state["info"], handle)
    rospy.spin()


if __name__ == "__main__":
    main()
