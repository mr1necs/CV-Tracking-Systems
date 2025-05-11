import logging
import time
from argparse import ArgumentParser
from collections import deque

import cv2
import imutils
import numpy as np
from torch import cuda
from torch.backends import mps
from ultralytics import YOLO

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("-d", "--device", type=str, default='cpu', help="Device: 'mps', 'cuda' or 'cpu'")
    ap.add_argument("-c", "--camera", type=str, default=None, help="Path to the optional video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="Maximum buffer size for trajectory")
    return vars(ap.parse_args())


def get_model(device_preference='cpu'):
    device = (
        'mps' if device_preference == 'mps' and mps.is_available() else
        'cuda' if device_preference == 'cuda' and cuda.is_available() else
        'cpu'
    )

    try:
        model = YOLO('yolo11n.pt').to(device)
        logging.info(f"Model successfully loaded, selected device: {device}")
        return model

    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")


def get_video(video_path=None):
    camera = cv2.VideoCapture(0 if video_path is None else video_path)
    if not camera.isOpened(): raise RuntimeError("Error: Unable to open video stream")
    logging.info("Video stream successfully opened.")
    return camera


def process_frame_segment(data):
    model, frame_segment = data
    detected_objects = []
    start_time = time.time()

    for r in model(frame_segment):
        for det in r.boxes:
            conf, cls = det.conf.item(), int(det.cls.item())
            class_name = model.names[cls].lower()

            if class_name.lower() in ('frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock') and conf >= 0.3:
                detected_objects.append((class_name, conf, det.xyxy[0].cpu().numpy()))

    logging.debug(f"Processing completed. Time: {time.time() - start_time:.4f} seconds.")
    return detected_objects


def draw_tracking(frame, merged_detections, pts, buffer_size):
    start_time = time.time()

    for class_name, conf, (x1, y1, x2, y2) in merged_detections:
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        radius = int(abs((y1 - y2) / 2))
        pts.appendleft(center)
        cv2.circle(frame, center, radius, (0, 255, 255), 3)

    for i in range(1, len(pts)):
        if pts[i - 1] is not None and pts[i] is not None:
            thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Object Tracking", frame)
    logging.debug(f"Object rendering on screen completed. Time: {time.time() - start_time:.4f} seconds.")

    return frame


if __name__ == "__main__":
    args = get_arguments()
    camera = get_video(args["camera"])
    neural = get_model(args["device"])
    pts = deque(maxlen=args["buffer"])

    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        start_time = time.time()
        grabbed, frame = camera.read()
        frame = imutils.resize(frame, width=1080)
        if not grabbed: break
        logging.debug(f"Image capture and preprocessing completed. Time: {time.time() - start_time:.4f} seconds.")
        frame = draw_tracking(frame, process_frame_segment([neural, frame]), pts, args["buffer"])
        logging.info(f"Frame processing fully completed. Time: {time.time() - start_time:.4f} seconds.\n\n")

    logging.info("Program terminated by user request.")
    camera.release()
    cv2.destroyAllWindows()
