import sys, os, shutil
from time import sleep
from A2C_AGENT import a2c_agent
from A2C_MODEL import a2c_model
from A2C_SERIAL import a2c_serial


def file_backup(log_dir):
    file_name_list = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if '.py' in x]
    file_path_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)),x) for x in file_name_list if '.py' in x]
    for fname, fpath in zip(file_name_list, file_path_list):
        shutil.copy(src=fpath, dst=os.path.join(log_dir,fname))

INIT_MESSAGE = '''
using acrobot-v2 environment which is d2h10s edition v4.0
definition of reward : [reward = |sin(theta_1)|]
termination condition: FFT
''' # 1/|cos(theta1)+0.1|-1/(1+0.1)

_load_dir = sys.argv[1] if len(sys.argv) > 1 else ""
#_load_dir = 'Acrobot-v2_0819_15-38-25_test'
env = a2c_serial()
while not env.serial_open(target_port='COM8'):
    sleep(0.5)

if __name__ == '__main__' and env.ser.isOpen():
    observation_n = env.observation_space_n
    hidden_n = 128
    action_n = env.action_space_n

    model = a2c_model(observation_n, hidden_n, action_n, load_dir=_load_dir)
    agent = a2c_agent(model, lr="12e-5", sampling_time=0.04, version='v4', suffix="r1")
    agent.init_message(INIT_MESSAGE)
    file_backup(agent.log_dir)

    agent.train(env)

    env.serial_close()