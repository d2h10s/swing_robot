from A2C_SERIAL import a2c_serial
import numpy as np
import time

ser = a2c_serial()

ser.serial_open()
while True:
    th1, th2, vel1, vel2 = ser.get_observation()
    print(np.rad2deg(th1))
    time.sleep(0.01)