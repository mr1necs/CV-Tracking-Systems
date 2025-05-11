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
STEP_DELAY = 0.0001         # delay between steps, seconds
MAX_STEPS = 100             # max steps per control command
HORIZ_THRESHOLD = 50        # pixels

# — PID controller gains —
PID_KP = 0.0335
PID_KI = 0.0185
PID_KD = 0.00075

# — Camera and detection settings —
DEVICE_PREFERENCE = 'cpu'   # 'cpu', 'cuda' or 'mps'
CAMERA_SOURCE = None        # pass index or URL
CONFIDENCE_THRESHOLD = 0.7
DISPLAY_WINDOW = True

# — Logging setup —
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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
        if self.max_out is not None and output > self.max_out:
            output = self.max_out
        if self.min_out is not None and output < self.min_out:
            output = self.min_out
        logging.debug(f"PID update — P: {P:.3f}, I: {I:.3f}, D: {D:.3f}, output: {output:.3f}")
        return output


def select_device(preference: str) -> str:
    """Choose compute device for YOLO."""
    if preference == 'mps' and mps.is_available(): return 'mps'
    elif preference == 'cuda' and cuda.is_available(): return 'cuda'
    return 'cpu'


def load_yolo_model(device: str):
    """Load YOLO model onto specified device."""
    model = YOLO('yolo11n.pt').to(device)
    logging.info(f"YOLO model loaded on {device}")
    return model


def open_camera(source):
    """Open video capture."""
    cam = cv2.VideoCapture(0 if source is None else source)
    if not cam.isOpened(): raise RuntimeError("Failed to open video stream")
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
            logging.debug(f"Detected {cls_name} at ({cx}, {cy}) with confidence {conf:.2f}")
            return cx, cy, x2 - x1, y2 - y1
    return None


def init_tracker(frame, bbox):
    """Initialize KCF tracker with given bbox."""
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, bbox)
    logging.debug(f"Tracker initialized at bbox {bbox}")
    return tracker


if __name__ == '__main__':
    try:
        device = select_device(DEVICE_PREFERENCE)
        model = load_yolo_model(device)
        camera = open_camera(CAMERA_SOURCE)
        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)

        pid = PIDController(kp=PID_KP, ki=PID_KI, kd=PID_KD, dt=0.04, output_limits=(-MAX_STEPS // 2, MAX_STEPS // 2))
        logging.debug("PID controller initialized")

        tracker = None
        frame_counter = 0

        # enable motors (LOW = enabled)
        GPIO.output(PAN_ENABLE_PIN, GPIO.LOW)
        GPIO.output(TILT_ENABLE_PIN, GPIO.LOW)

        while True:
            ret, frame = camera.read()
            if not ret:
                logging.error("Failed to read frame from camera")
                break

            frame = imutils.resize(frame, width=800)
            height, width = frame.shape[:2]
            frame_counter += 1
            restart_detection = (frame_counter % 30 == 0)

            # Detection or tracking
            if tracker is None or restart_detection:
                detection = detect_object(model, frame, CONFIDENCE_THRESHOLD)
                frame_counter = 0
                if detection:
                    cx, cy, bw, bh = detection
                    bbox = (cx - bw // 2, cy - bh // 2, bw, bh)
                    tracker = init_tracker(frame, bbox)
                    pid.integral = pid.prev_error = 0.0
            else:
                ok, bbox = tracker.update(frame)
                if not ok:
                    logging.warning("Tracker lost object")
                    tracker = None
                    continue
                x, y, bw, bh = map(int, bbox)
                cx = x + bw // 2
                if DISPLAY_WINDOW:
                    cv2.circle(frame, (cx, y + bh // 2), 5, (0, 255, 0), -1)

            # Motor control based on horizontal error
            if tracker:
                error_x = cx - width / 2
                if abs(error_x) > HORIZ_THRESHOLD:
                    control_signal = pid.update(error_x)
                    steps_to_move = max(1, int(abs(control_signal)))
                    direction_cw = control_signal < 0
                    logging.debug(f"Control signal: {control_signal:.2f}, steps: {steps_to_move}, {'CW' if direction_cw else 'CCW'}")

                    t_pan = threading.Thread(target=move_motor, args=(pan_motor, direction_cw, steps_to_move))
                    t_tilt = threading.Thread(target=move_motor, args=(tilt_motor, direction_cw, steps_to_move))
                    t_pan.start()
                    t_tilt.start()
                    t_pan.join()
                    t_tilt.join()

            if DISPLAY_WINDOW: cv2.imshow('Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Exit key pressed")
                break

    finally:
        GPIO.output(PAN_ENABLE_PIN, GPIO.HIGH)
        GPIO.output(TILT_ENABLE_PIN, GPIO.HIGH)
        GPIO.cleanup()
        camera.release()
        cv2.destroyAllWindows()
        logging.info("Cleaned up GPIO and closed windows")
