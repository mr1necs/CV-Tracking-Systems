import time
import threading
import RPi.GPIO as GPIO
from RpiMotorLib import RpiMotorLib


DIR1_PIN, STEP1_PIN, EN1_PIN = 23, 24, 25
DIR2_PIN, STEP2_PIN, EN2_PIN = 17, 27, 22
MS_PINS = (21, 21, 21)
DELAY = 0.0001
STEPS = 1600
TURNS = 1

pan_motor = RpiMotorLib.A4988Nema(DIR1_PIN, STEP1_PIN, MS_PINS, "DRV8825")
til_motor = RpiMotorLib.A4988Nema(DIR2_PIN, STEP2_PIN, MS_PINS, "DRV8825")
GPIO.setup(EN1_PIN, GPIO.OUT)
GPIO.setup(EN2_PIN, GPIO.OUT)


def move_motor(motor, clockwise, steps):
    motor.motor_go(clockwise=clockwise, steptype="1/8", steps=steps, stepdelay=DELAY, verbose=True, initdelay=0.05)


try:
    GPIO.output(EN1_PIN, GPIO.LOW)
    GPIO.output(EN2_PIN, GPIO.LOW)
    turnover = int(STEPS * TURNS)

    t1 = threading.Thread(target=move_motor, args=(pan_motor, True, turnover))
    t2 = threading.Thread(target=move_motor, args=(til_motor, True, turnover))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    time.sleep(1)

    t3 = threading.Thread(target=move_motor, args=(pan_motor, False, turnover))
    t4 = threading.Thread(target=move_motor, args=(til_motor, False, turnover))
    t3.start()
    t4.start()
    t3.join()
    t4.join()

finally:
    GPIO.output(EN1_PIN, GPIO.HIGH)
    GPIO.output(EN2_PIN, GPIO.HIGH)
    GPIO.cleanup()
