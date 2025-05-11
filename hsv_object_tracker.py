from argparse import ArgumentParser
from collections import deque

import cv2
import imutils
import logging
import numpy as np
import time

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def get_arguments():
    ap = ArgumentParser()
    ap.add_argument("-c", "--camera", type=str, default=None, help="Path to an optional video file")
    ap.add_argument("-b", "--buffer", type=int, default=64, help="Maximum buffer size for trajectory tracking")
    return vars(ap.parse_args())


def get_video(video_path=None):
    camera = cv2.VideoCapture(0 if video_path is None else video_path)
    if not camera.isOpened(): raise RuntimeError("Error: Unable to open video stream")
    logging.info("Video stream successfully opened.")
    return camera


def process_frame(frame, color_lower, color_upper):
    start_time = time.time()
    frame = imutils.resize(frame, width=1080)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    logging.debug(f"Processing completed. Time: {time.time() - start_time:.4f} seconds.")
    return mask, frame


def find_and_draw_contours(frame, mask, pts, buffer_size):
    start_time = time.time()
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        moments = cv2.moments(largest_contour)
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))

        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is not None and pts[i] is not None:
            thickness = int(np.sqrt(buffer_size / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("HSV-Based Object Tracking", frame)
    logging.debug(f"Object rendering on screen completed. Time: {time.time() - start_time:.4f} seconds.")
    return frame


if __name__ == "__main__":
    args = get_arguments()
    camera = get_video(args["camera"])
    pts = deque(maxlen=args["buffer"])
    green_lower, green_upper = (29, 86, 6), (64, 255, 255)

    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        start_time = time.time()
        grabbed, frame = camera.read()
        if not grabbed: break
        logging.debug(f"Image capture and preprocessing completed. Time: {time.time() - start_time:.4f} seconds.")
        mask, processed_frame = process_frame(frame, green_lower, green_upper)
        output_frame = find_and_draw_contours(processed_frame, mask, pts, args["buffer"])
        logging.info(f"Frame processing fully completed. Time: {time.time() - start_time:.4f} seconds.\n\n")

    logging.info("Program terminated by user request.")
    camera.release()
    cv2.destroyAllWindows()