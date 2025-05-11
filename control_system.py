import logging
import threading

import RPi.GPIO as GPIO
import cv2
import imutils
from RpiMotorLib import RpiMotorLib
from torch import cuda
from torch.backends import mps
from ultralytics import YOLO

# — GPIO pins for the stepper motors —
PAN_DIR_PIN, PAN_STEP_PIN, PAN_ENABLE_PIN = 23, 24, 25
TILT_DIR_PIN, TILT_STEP_PIN, TILT_ENABLE_PIN = 17, 27, 22
MICROSTEP_PINS = (21, 21, 21)

# — Motion parameters —
STEP_DELAY = 0.0001         # delay between microsteps, seconds
MAX_STEPS = 100             # max steps per control command
HORIZ_THRESHOLD = 50        # horizontal error threshold in pixels
VERT_THRESHOLD = 50         # vertical error threshold in pixels

# — PID controller gains —
PAN_PID_KP = 0.0335
PAN_PID_KI = 0.0185
PAN_PID_KD = 0.00075
TILT_PID_KP = 0.0335
TILT_PID_KI = 0.0185
TILT_PID_KD = 0.00075

# — Camera and detection settings —
DEVICE_PREFERENCE = 'cpu'       # 'cpu', 'cuda' or 'mps'
CAMERA_SOURCE = None            # camera index or stream URL
CONFIDENCE_THRESHOLD = 0.7
DISPLAY_WINDOW = True

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# — Initialize motors —
pan_motor = RpiMotorLib.A4988Nema(PAN_DIR_PIN, PAN_STEP_PIN, MICROSTEP_PINS, "DRV8825")
tilt_motor = RpiMotorLib.A4988Nema(TILT_DIR_PIN, TILT_STEP_PIN, MICROSTEP_PINS, "DRV8825")

GPIO.setmode(GPIO.BCM)
GPIO.setup(PAN_ENABLE_PIN, GPIO.OUT)
GPIO.setup(TILT_ENABLE_PIN, GPIO.OUT)


def move_motor(motor, clockwise: bool, steps: int):
    """Drive one motor for a given number of microsteps."""
    logging.debug(f"Moving motor {'CW' if clockwise else 'CCW'} for {steps} steps")
    motor.motor_go(
        clockwise=clockwise,
        steptype="1/8",
        steps=steps,
        stepdelay=STEP_DELAY,
        verbose=False,
        initdelay=0.04
    )


class PIDController:
    def __init__(self, kp, ki, kd, dt=0.02, output_limits=(None, None)):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.min_out, self.max_out = output_limits

    def update(self, error):
        """Compute PID control signal."""
        self.integral += error * self.dt
        P = self.kp * error
        I = self.ki * self.integral
        D = self.kd * (error - self.prev_error) / self.dt
        self.prev_error = error

        output = P + I + D
        if self.max_out is not None and output > self.max_out: output = self.max_out
        if self.min_out is not None and output < self.min_out: output = self.min_out
        logging.debug(f"PID update — P: {P:.3f}, I: {I:.3f}, D: {D:.3f}, output: {output:.3f}")
        return output


def select_device(preference: str) -> str:
    """Choose compute device for YOLO."""
    if preference == 'mps' and mps.is_available():
        device = 'mps'
    elif preference == 'cuda' and cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info(f"Selected device: {device}")
    return device


def load_yolo_model(device: str):
    """Load YOLO model onto specified device."""
    model = YOLO('yolo11n.pt').to(device)
    logging.info(f"YOLO model loaded on {device}")
    return model


def open_camera(source):
    """Open video capture."""
    cam = cv2.VideoCapture(0 if source is None else source)
    if not cam.isOpened():
        raise RuntimeError("Failed to open video stream")
    logging.info("Video stream opened")
    return cam


def detect_object(model, frame, conf_thresh):
    """Run one detection pass, return center and bbox size if found."""
    results = model(frame)
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls)].lower()
        conf = float(box.conf)
        if cls_name in ('frisbee', 'sports ball', 'apple', 'orange', 'cake', 'clock') and conf >= conf_thresh:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            logging.info(f"Detected {cls_name} at ({cx}, {cy}) with confidence {conf:.2f}")
            return cx, cy, x2 - x1, y2 - y1
    return None


def init_tracker(frame, bbox):
    """Initialize KCF tracker with given bbox."""
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, bbox)
    logging.info(f"Tracker initialized at bbox {bbox}")
    return tracker


if __name__ == '__main__':
    try:
        device = select_device(DEVICE_PREFERENCE)
        model = load_yolo_model(device)
        video_capture = open_camera(CAMERA_SOURCE)
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)

        # initialize separate PID controllers for pan (horizontal) and tilt (vertical)
        pid_pan = PIDController(kp=PAN_PID_KP, ki=PAN_PID_KI, kd=PAN_PID_KD, dt=0.04,
                                output_limits=(-MAX_STEPS // 2, MAX_STEPS // 2))
        pid_tilt = PIDController(kp=TILT_PID_KP, ki=TILT_PID_KI, kd=TILT_PID_KD, dt=0.04,
                                 output_limits=(-MAX_STEPS // 2, MAX_STEPS // 2))
        logging.info("PID controllers for pan and tilt initialized")

        tracker = None
        frame_count = 0

        # enable motors (LOW = enabled)
        GPIO.output(PAN_ENABLE_PIN, GPIO.LOW)
        GPIO.output(TILT_ENABLE_PIN, GPIO.LOW)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                break

            frame = imutils.resize(frame, width=800)
            height, width = frame.shape[:2]

            frame_count += 1
            force_detection = (frame_count % 30 == 0)

            if tracker is None or force_detection:
                detection = detect_object(model, frame, CONFIDENCE_THRESHOLD)
                frame_count = 0
                if detection:
                    cx, cy, bw, bh = detection
                    bbox = (cx - bw // 2, cy - bh // 2, bw, bh)
                    tracker = init_tracker(frame, bbox)
                    # reset both PIDs
                    pid_pan.integral = pid_pan.prev_error = 0.0
                    pid_tilt.integral = pid_tilt.prev_error = 0.0
            else:
                ok, bbox = tracker.update(frame)
                if not ok:
                    logging.warning("Tracker lost object")
                    tracker = None
                    continue
                x, y, bw, bh = map(int, bbox)
                cx, cy = x + bw // 2, y + bh // 2
                if DISPLAY_WINDOW:
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            # control both motors independently
            if tracker:
                error_x = cx - width / 2
                error_y = cy - height / 2

                pan_steps, tilt_steps = 0, 0
                pan_dir, tilt_dir = False, False

                if abs(error_x) > HORIZ_THRESHOLD:
                    control_x = pid_pan.update(error_x)
                    pan_steps = max(1, int(abs(control_x)))
                    pan_dir = control_x < 0
                    logging.info(
                        f"Pan control — error: {error_x:.2f}, signal: {control_x:.2f}, steps: {pan_steps}, {'CW' if pan_dir else 'CCW'}")

                if abs(error_y) > VERT_THRESHOLD:
                    control_y = pid_tilt.update(error_y)
                    tilt_steps = max(1, int(abs(control_y)))
                    tilt_dir = control_y < 0
                    logging.info(f"Tilt control — error: {error_y:.2f}, signal: {control_y:.2f}, steps: {tilt_steps}, {'CW' if tilt_dir else 'CCW'}")

                threads = []
                if pan_steps: threads.append(threading.Thread(target=move_motor, args=(pan_motor, pan_dir, pan_steps)))
                if tilt_steps: threads.append(threading.Thread(target=move_motor, args=(tilt_motor, tilt_dir, tilt_steps)))
                for t in threads: t.start()
                for t in threads: t.join()

            if DISPLAY_WINDOW:
                cv2.imshow('Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exit key pressed, shutting down")
                    break

    finally:
        GPIO.output(PAN_ENABLE_PIN, GPIO.HIGH)
        GPIO.output(TILT_ENABLE_PIN, GPIO.HIGH)
        GPIO.cleanup()
        video_capture.release()
        cv2.destroyAllWindows()
        logging.info("Cleaned up GPIO and closed windows")
