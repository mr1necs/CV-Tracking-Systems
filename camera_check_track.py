import logging
import time
from collections import deque

import cv2
import imutils
from torch import cuda
from torch.backends import mps
from ultralytics import YOLO

DEVICE_PREF = 'cpu'
CAMERA_SRC = None
BUFFER_SIZE = 64
CONF_THRESH = 0.3
DISPLAY = True

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def select_device(pref):
    if pref == 'mps' and mps.is_available(): return 'mps'
    if pref == 'cuda' and cuda.is_available(): return 'cuda'
    return 'cpu'


def load_model(device):
    try:
        model_ = YOLO('yolo11n.pt').to(device)
        logging.info(f"YOLO model loaded on {device}")
        return model_
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {e}")


def open_stream(source):
    camera = cv2.VideoCapture(0 if source is None else source)
    if not camera.isOpened(): raise RuntimeError("Failed to open video stream")
    logging.info("Video stream opened")
    return camera


def detect_object(model_, window, conf_thresh):
    for result in model_(window):
        for box in result.boxes:
            conf = float(box.conf)
            name = model_.names[int(box.cls)].lower()
            if name in ('frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock') and conf >= conf_thresh:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                return name, (x1, y1, x2 - x1, y2 - y1)
    return None


def init_tracker(window, box):
    track = cv2.legacy.TrackerCSRT_create()
    track.init(window, box)
    return track


def draw(window, points, radius):
    cv2.circle(window, points[0], radius, (0, 255, 255), 3)
    for i in range(1, len(points)):
        if points[i - 1] and points[i]:
            thickness = int((len(points) / (i + 1)) ** 0.5 * 2.5)
            cv2.line(window, points[i - 1], points[i], (0, 0, 255), thickness)


if __name__ == '__main__':
    device = select_device(DEVICE_PREF)
    model = load_model(device)
    cap = open_stream(CAMERA_SRC)

    cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)

    tracker = None
    current_cls = None
    pts = deque(maxlen=BUFFER_SIZE)

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = imutils.resize(frame, width=480)
        start = time.time()
        frame_counter += 1
        force_detect = frame_counter % 30 == 0

        if tracker is None or force_detect:
            det = detect_object(model, frame, CONF_THRESH)
            if det:
                current_cls, bbox = det
                tracker = init_tracker(frame, bbox)
                logging.info(f"Detector {'reinitialized' if force_detect else 'initialized'} tracker for {current_cls}")
			frame_counter = 1    
        else:
            ok, bbox = tracker.update(frame)
            if not ok:
                logging.warning("Tracker lost the object")
                tracker = None
                current_cls = None
            else:
                x, y, w, h = map(int, bbox)
                pts.appendleft((x + w // 2, y + h // 2))
                if DISPLAY: draw(frame, pts, h // 2)

        logging.debug(f"Frame time {frame_counter}: {time.time() - start:.3f} sec")

        if DISPLAY: cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
