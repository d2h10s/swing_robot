import sys, os, shutil
from time import sleep
from A2C_AGENT import a2c_agent
from A2C_MODEL import a2c_model
from A2C_SERIAL import a2c_serial

# backup code files to log folder when start program
def file_backup(log_dir):
    file_name_list = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if '.py' in x]
    file_path_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)),x) for x in file_name_list if '.py' in x]
    for fname, fpath in zip(file_name_list, file_path_list):
        shutil.copy(src=fpath, dst=os.path.join(log_dir,fname))

INIT_MESSAGE = \
'''using acrobot-v2 environment which is d2h10s edition v4.0
definition of reward : [reward = |sin(theta_1)|]
termination condition: FFT
'''

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
    agent = a2c_agent(model, lr="12e-5", sampling_time=0.04, version='v5', suffix="r1", nstart=False, test=True)
    agent.run_test(env)
    env.serial_close()