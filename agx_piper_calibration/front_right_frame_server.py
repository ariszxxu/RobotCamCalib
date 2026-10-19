"""Publish front and right-wrist color frames into shared memory.

This is a read-only ROS subscriber. It does not open either camera device,
publish actuator targets, or interact with upload processes.
"""
import json
import os
import struct
import time

import rospy
from sensor_msgs.msg import CameraInfo, Image

CAMERAS = {
    "front": {
        "namespace": "camera_f",
        "shm": "/dev/shm/front-d435-color.bin",
        "info": "/dev/shm/front-d435-color-info.json",
        "magic": b"FSD435\0\0",
    },
    "right_wrist": {
        "namespace": "camera_r",
        "shm": "/dev/shm/right-d435-color.bin",
        "info": "/dev/shm/right-d435-color-info.json",
        "magic": b"RSD435\0\0",
    },
}


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
    rospy.init_node("front_right_cv2_source", anonymous=True, disable_signals=True)
    state = {
        role: {"info": None, "seq": 0}
        for role in CAMERAS
    }

    def on_info(msg, role):
        state[role]["info"] = {
            "width": msg.width,
            "height": msg.height,
            "stamp": msg.header.stamp.to_sec(),
            "frame_id": msg.header.frame_id,
            "distortion_model": msg.distortion_model,
            "D": list(msg.D),
            "K": list(msg.K),
        }

    def on_image(msg, role):
        spec = CAMERAS[role]
        image = bgr_array(msg)
        state[role]["seq"] += 1
        header = struct.pack(
            "<8sQdIII",
            spec["magic"],
            state[role]["seq"],
            msg.header.stamp.to_sec(),
            image.shape[0],
            image.shape[1],
            3,
        )
        temporary = "%s.%d.tmp" % (spec["shm"], os.getpid())
        with open(temporary, "wb") as handle:
            handle.write(header + image.tobytes())
        os.replace(temporary, spec["shm"])

    for role, spec in CAMERAS.items():
        root = "/" + spec["namespace"] + "/color/"
        rospy.Subscriber(
            root + "camera_info",
            CameraInfo,
            on_info,
            callback_args=role,
            queue_size=1,
        )
        rospy.Subscriber(
            root + "image_raw",
            Image,
            on_image,
            callback_args=role,
            queue_size=1,
            buff_size=8000000,
        )

    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline and not rospy.is_shutdown():
        if all(state[role]["info"] is not None for role in CAMERAS):
            break
        time.sleep(0.05)
    missing = [role for role in CAMERAS if state[role]["info"] is None]
    if missing:
        raise RuntimeError("Missing camera_info: " + ", ".join(missing))
    for role, spec in CAMERAS.items():
        with open(spec["info"], "w") as handle:
            json.dump(state[role]["info"], handle)
    rospy.spin()


if __name__ == "__main__":
    main()
