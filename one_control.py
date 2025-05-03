import time
import logging
from collections import deque

import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import cv2
import imutils
import numpy as np

# ── CONFIGURATION ───────────────────────────────────────────────────────────────

DIR, STEP, EN = 22, 23, 24
MS_PINS = (21, 20, 16)

STEPS_PER_REV = 200
MAX_ANGLE_DEG = 120
STEPS_PER_DEG = STEPS_PER_REV / 360.0
MAX_STEPS = int(MAX_ANGLE_DEG * STEPS_PER_DEG)

FRAME_WIDTH = 1080
CAM_INDEX = 0
COLOR_LOWER = tuple[29, 86, 6]
COLOR_UPPER = tuple[64, 255, 255]
BUFFER_SIZE = 64

SCAN_STEP_DEG = 2
SCAN_STEP = int(SCAN_STEP_DEG * STEPS_PER_DEG)
SCAN_DELAY = 0.01

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ── SETUP ──────────────────────────────────────────────────────────────────────

GPIO.setmode(GPIO.BCM)
GPIO.setup(EN, GPIO.OUT)
GPIO.output(EN, GPIO.LOW)
motor = RpiMotorLib.A4988Nema(DIR, STEP, MS_PINS, "DRV8825")


cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Unable to open camera!")

pts = deque(maxlen=BUFFER_SIZE)

current_steps = 0
scan_dir = 1

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("Frame grab failed, exiting.")
            break

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, COLOR_LOWER, COLOR_UPPER)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            if M["m00"] > 0 and radius > 10:
                center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

        pts.appendleft(center)
        for i in range(1, len(pts)):
            if pts[i-1] and pts[i]:
                thickness = int(
                    np.sqrt(BUFFER_SIZE/float(i+1))*2.5
                )
                cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), thickness)

        # ── OBJECT TRACKING OR SCANNING ──────────────────────────
        if center:
            err_px = center[0] - (FRAME_WIDTH // 2)
            norm = err_px / float(FRAME_WIDTH // 2)  # -1..+1
            desired_angle = (norm + 1)/2 * MAX_ANGLE_DEG
            desired_steps = int(desired_angle * STEPS_PER_DEG)

            delta = desired_steps - current_steps
            if delta != 0:
                cw = delta > 0
                motor.motor_go(clockwise=not cw, steptype="Full", steps=abs(delta), stepdelay=0.001, verbose=False, initdelay=0.05)
                current_steps = desired_steps

        else:
            if current_steps >= MAX_STEPS and scan_dir > 0:
                scan_dir = -1
            elif current_steps <= 0 and scan_dir < 0:
                scan_dir = 1

            motor.motor_go(
                clockwise=(scan_dir > 0),
                steptype="Full",
                steps=SCAN_STEP,
                stepdelay=SCAN_DELAY,
                verbose=False,
                initdelay=0.05
            )
            current_steps += scan_dir * SCAN_STEP
            current_steps = max(0, min(MAX_STEPS, current_steps))

        cv2.imshow("Tracking + Pan Control", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    logging.info("Cleaning up GPIO and camera.")
    cap.release()
    cv2.destroyAllWindows()
    GPIO.output(EN, GPIO.HIGH)
    GPIO.cleanup()
