from contaccams.cameras import CameraManager
import cv2
import numpy as np
import time
from math import ceil, sqrt

# --- helpers ---
def _fit_size_preserve_aspect(src_w, src_h, dst_w, dst_h):
    """Return new size (w, h) fitting into dst while preserving aspect ratio."""
    if src_w == 0 or src_h == 0:
        return dst_w, dst_h
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))

def _put_label(img, text, margin=6):
    """Draw a filled rect + text at top-left for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.6
    th = 2
    (tw, th_pix), _ = cv2.getTextSize(text, font, fs, th)
    x1, y1 = 5, 5
    x2, y2 = x1 + tw + margin*2, y1 + th_pix + margin*2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.putText(img, text, (x1 + margin, y2 - margin), font, fs, (255, 255, 255), th, cv2.LINE_AA)

def _tile_frames(name_to_frame, window_wh=(1600, 900), padding=8, bg_color=(30, 30, 30)):
    """
    Tile frames into a single canvas sized to window_wh.
    Returns: tiled_canvas (H, W, 3)
    """
    W, H = window_wh
    names = list(name_to_frame.keys())
    frames = [name_to_frame[k] for k in names if name_to_frame[k] is not None]
    names  = [k for k in names if name_to_frame[k] is not None]
    n = len(frames)

    if n == 0:
        # empty canvas placeholder
        canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)
        _put_label(canvas, "No frames available")
        return canvas

    # choose grid cols/rows
    cols = ceil(sqrt(n))
    rows = ceil(n / cols)

    # compute tile area with padding around and between tiles
    total_pad_x = padding * (cols + 1)
    total_pad_y = padding * (rows + 1)
    tile_w = max(1, (W - total_pad_x) // cols)
    tile_h = max(1, (H - total_pad_y) // rows)

    canvas = np.full((H, W, 3), bg_color, dtype=np.uint8)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n:
                break
            frame = frames[idx]
            name  = names[idx]
            h, w = frame.shape[:2]

            # fit into tile while keeping aspect
            new_w, new_h = _fit_size_preserve_aspect(w, h, tile_w, tile_h)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # put label
            _put_label(resized, name)

            # compute top-left with centering inside its tile
            x0 = padding + c * (tile_w + padding) + (tile_w - new_w) // 2
            y0 = padding + r * (tile_h + padding) + (tile_h - new_h) // 2

            # paste
            canvas[y0:y0+new_h, x0:x0+new_w] = resized
            idx += 1

    return canvas

# --- main visualization loop ---
def visualize_all_cameras(manager, window_name="Cameras", window_wh=(1400, 800), convert_rgb=False):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_wh[0], window_wh[1])

    while True:
        frames_dict = manager.read_all_images(convert_rgb=convert_rgb)
        canvas = _tile_frames(frames_dict, window_wh=window_wh)
        cv2.imshow(window_name, canvas)

        # Single, short wait for GUI events; responsive to 'q'/'Q'/Esc
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # 27 = Esc
            break

        # Also exit if user closes the window
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    camera_to_port = {
        "tip_cam": "3-10:1.0",
        "root_cam": "3-9:1.0",
    }

    manager = CameraManager(camera_to_port)

    # Open all cameras
    opened_count = manager.open_all_cameras()
    print(f"Successfully opened {opened_count} cameras")

    # Live visualize in a single window (press 'q' to quit)
    visualize_all_cameras(manager, window_wh=(1400, 800), convert_rgb=False)

    # Release all cameras when done
    manager.release_all()

