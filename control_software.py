import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib
import cv2
import numpy as np
import imutils
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

GPIO.setmode(GPIO.BCM)
DIR1, STEP1, EN1 = 22, 23, 24
DIR2, STEP2, EN2 = 17, 27, 25
MS_PINS = (21, 21, 21)

motor_pan = RpiMotorLib.A4988Nema(DIR1, STEP1, MS_PINS, "DRV8825")
motor_tilt = RpiMotorLib.A4988Nema(DIR2, STEP2, MS_PINS, "DRV8825")

GPIO.setup(EN1, GPIO.OUT)
GPIO.setup(EN2, GPIO.OUT)
GPIO.output(EN1, GPIO.LOW)
GPIO.output(EN2, GPIO.LOW)

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

PIXEL_THRESHOLD = 50
current_pan = 0.0
current_tilt = 60.0
last_pan_dir = True
green_lower, green_upper = (29, 86, 6), (64, 255, 255)

def process_frame(frame):
    frame = imutils.resize(frame, width=1080)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask, frame

def detect_and_draw(mask, frame):
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0 and radius > 10:
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
    cv2.imshow("Tracking", frame)
    return center

def move_pan(steps, clockwise):
    global current_pan
    motor_pan.motor_go(clockwise=clockwise, step_type="Full", steps=steps, step_delay=0.001, verbose=False, init_delay=0.05)
    delta = steps * 1.8 * (1 if clockwise else -1)
    current_pan = (current_pan + delta) % 360

def move_tilt(steps, clockwise):
    global current_tilt
    motor_tilt.motor_go(clockwise=clockwise, step_type="Full", steps=steps, step_delay=0.001, verbose=False, init_delay=0.05)
    delta = steps * 1.8 * (1 if clockwise else -1)
    new = current_tilt + delta
    if 0 <= new <= 120:
        current_tilt = new

def track(center, frame):
    global last_pan_dir
    h, w = frame.shape[:2]
    cx, cy = w//2, h//2
    dx = center[0] - cx
    dy = center[1] - cy
    if abs(dx) > PIXEL_THRESHOLD:
        dir_pan = dx > 0
        move_pan(1, dir_pan)
        last_pan_dir = dir_pan
    if abs(dy) > PIXEL_THRESHOLD:
        dir_tilt = dy > 0
        move_tilt(1, dir_tilt)

def search():
    move_pan(1, last_pan_dir)

try:
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        mask, vis = process_frame(frame)
        center = detect_and_draw(mask, vis)
        if center:
            track(center, vis)
        else:
            search()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    GPIO.output(EN1, GPIO.HIGH)
    GPIO.output(EN2, GPIO.HIGH)
    GPIO.cleanup()
    camera.release()
    cv2.destroyAllWindows()
