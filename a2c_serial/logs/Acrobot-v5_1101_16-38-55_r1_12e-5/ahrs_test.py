from A2C_SERIAL import a2c_serial
import numpy as np
import time

ser = a2c_serial()

ser.serial_open()
while True:
    state = ser.get_observation()
    th1, th2, vel1, vel2 = state
    th1, th2 = np.rad2deg(th1), np.rad2deg(th2)
    print(th1, th2, vel1, vel2)
    time.sleep(0.1)