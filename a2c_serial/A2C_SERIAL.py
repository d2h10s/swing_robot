import sys, serial, glob, time
import numpy as np
from time import sleep

DEBUG_ON = False

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

ACQ = b'\x04'
RST = b'\x05'
GO_CW = b'\x70'
GO_CCW = b'\x71'

DEBUGGING = 1
COMMAND = 2

BAUDRATE = 115200

class a2c_serial:
    def __init__(self):
        self.ser = serial.Serial()
        self.port = None
        self.observation_space_n = 4
        self.action_space_n = 2
        self.wait_time = 190
        self.max_angle = 0
        self.zero_angle = 0
        self.EPS = np.finfo(np.float32).eps.item()
    
    def serial_open(self, target_port=None):
        if target_port==None:
            if sys.platform.startswith('win'):
                ports = ['COM%s' % i for i in range(1, 255)]  # 1~257
            elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
                # this excludes your current terminal "/dev/tty"
                ports = glob.glob('/dev/tty[A-Za-z]*')
            elif sys.platform.startswith('darwin'):
                ports = glob.glob('/dev/tty.*')
            else:
                raise EnvironmentError('Unsupported platform')
        else:
            ports = [target_port]
        print('searching serial device...')
        for port in ports:
            try:
                self.ser = serial.Serial(port, baudrate=BAUDRATE, timeout=1, write_timeout=1)
                reply, data_type = self.write_command(RST)
                if reply.startswith('STX,ACK') and data_type == COMMAND:
                    print(f'serial {port} found')
                    self.port = port
                    self.ser.timeout = 3
                    self.ser.write_timeout = 3
                    return True
                else:
                    print(f'serial {port} found but NAK', reply)
                    self.ser.close()
            except Exception as e:
                print(port, e)
        print('could not find serial device\n')
        return False

    def serial_close(self):
        self.ser.close()

    def write_command(self, command):
        if DEBUG_ON: print('start write')
        ret = self.ser.isOpen()
        while ret:
            self.ser.write(command)
            rx_buffer = bytearray()
            rx_byte = None
            byte_cnt = 0
            start_time = time.time()
            while rx_byte != ord('!') and time.time()-start_time < 10 and byte_cnt < 128:
                if self.ser.in_waiting:
                    rx_byte = ord(self.ser.read())
                    rx_buffer.append(rx_byte)
                    byte_cnt += 1
            if rx_byte == None:
                print(f'{hex(ord(command))} sent but received null byte')
            try:
                rx_string = rx_buffer.decode('utf-8')
            except :
                return "", -1
            if rx_string.startswith('@'):
                rx_string = rx_string[:-1]
                data_type = 1
            elif rx_string.startswith('STX'):
                rx_string = rx_string[:-1]
                data_type = 2
            else:
                data_type = -1
                print('could not recognize data:', rx_buffer)
            if DEBUG_ON: print('end write')
            return rx_string, data_type

    def reset(self, n_episode):
        if DEBUG_ON: print('start reset')
        ret = self.ser.isOpen()
        self.ser.write(GO_CW)
        while self.ser.isOpen():
            reply, data_type = self.write_command(RST)
            if reply.startswith('STX,ACK') and data_type == COMMAND:
                start_time = time.time()
                elapsed_time = 0
                episode_delay = 10 if n_episode == 1 else 1.26*np.abs(self.max_angle)+81.8+15
                sleep(1)
                while elapsed_time < episode_delay:
                    elapsed_time = time.time() - start_time
                    print(f'\relapsed {elapsed_time:.2f}s of {episode_delay:.1f}s and completed {np.min([elapsed_time/episode_delay*100,100]):6.2f}%', end='')
                    sleep(0.99)
                if DEBUG_ON: print('end reset')
                obs = self.get_observation()
                self.max_angle = 0
                print(f'\n\nzero angle {np.rad2deg(obs[0]):.3f} deg')
                return obs
            else:
                print('received unrecognized bytes', reply)


    def set_zero_angle(self):
        self.get_observation()
        self.zero_angle = self.roll
        print(f'\n\nzero angle {np.rad2deg(self.zero_angle):.3f} deg')

    def step(self, action):
        if DEBUG_ON: print('start step')
        ret = self.ser.isOpen()
        while ret:
            try:
                if action == 0:  # action 0 is go up (clock wise)
                    self.ser.write(GO_CW)
                elif action == 1:       # action 1 is go down (counter clock wise)
                    self.ser.write(GO_CCW)
                else:
                    print('action is out of range', action)
                if DEBUG_ON: print('end step')
                return self.get_observation()
            except Exception as e:
                self.ser.close()
                print("write error occurred in step function", e)
                return

    def get_observation(self):
        if DEBUG_ON: print('start obs')
        ret = self.ser.isOpen()
        while ret:
            rx_data, data_type = self.write_command(ACQ)
            if data_type == COMMAND and rx_data.startswith('STX,ACQ'):
                try:
                    # STX,ACQ,ROLL,gyro_ahrs,POS_mx106,VEL_mx106
                    rx_data = [float(x) for x in rx_data.replace('STX,ACQ,', '').split(',')]
                    if any(np.isnan(rx_data)):
                        print('get nan value from opencm', rx_data)
                        continue
                    roll = np.deg2rad(rx_data[0])  # deg -> rad
                    ahrs_vel = np.deg2rad(rx_data[1])  # deg/s -> rad/s
                    mx106_pos = np.deg2rad((2100 - rx_data[2]) * 0.088) # deg -> rad
                    mx106_vel = rx_data[3] * np.pi / 30  # rpm -> rad/s
                    #print('{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f},{:6.2f}'.format(roll, ahrs_vel, ahrs_temp, mx106_pos, mx106_vel, mx106_temp))
                    ret = False
                except Exception as e:
                    print(e, ' occurred with', rx_data, 'in obs function')
            else:
                print('\rcould not recognize ', rx_data)
        self.roll = roll
        th1 = self.roll - self.zero_angle
        th2 = mx106_pos
        vel1 = ahrs_vel
        vel2 = mx106_vel
        self.max_angle = max(self.max_angle,np.abs(np.rad2deg(th1))) # deg

        observation = np.array([th1, th2, vel1, vel2], dtype=np.float32)
        if DEBUG_ON: print('end obs')
        return observation
        