import sys, os, shutil
from time import sleep
from A2C_AGENT import a2c_agent
from A2C_MODEL import a2c_model
from A2C_SERIAL import a2c_serial

'''
@argument1: path to load
@argument2: whether to rename directory name and time parameter to now time
'''
_load_dir = 'run_test'

'''
env can be gym environment or serial environment
you should change some codes in A2C_AGENT.py when use gym environment
so if you want to use gym environment you should operate codes from simulator directory
'''
env = a2c_serial()

# if you don't put parameter, a2c_serial class find serail port by auto
while not env.serial_open():
    sleep(0.5)

# when serial is not opened, error message will be printed
if __name__ == '__main__' and env.ser.isOpen():
    observation_n = env.observation_space_n
    hidden_n = 128
    action_n = env.action_space_n

    model = a2c_model(observation_n, hidden_n, action_n, load_dir=_load_dir)
    agent = a2c_agent(model, lr="12e-5", sampling_time=.04, version='v5', suffix="r1", nstart=False, test=True)
    agent.run_test(env, 1000)
    env.serial_close()