import sys, os, shutil
from time import sleep
from A2C_AGENT import a2c_agent
from A2C_MODEL import a2c_model
import gym

def file_backup(log_dir):
    file_name_list = [x for x in os.listdir(os.path.dirname(os.path.realpath(__file__))) if '.py' in x]
    file_path_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)),x) for x in file_name_list if '.py' in x]
    for fname, fpath in zip(file_name_list, file_path_list):
        shutil.copy(src=fpath, dst=os.path.join(log_dir,fname))

INIT_MESSAGE = '''
using acrobot-v2 environment which is d2h10s edition v3.0
definition of reward : [reward = 1/|cos(theta1)+0.1|-1/(1+0.1)]
termination condition: FFT
'''

_load_dir = sys.argv[1] if len(sys.argv) > 1 else ""
#_load_dir = 'Acrobot-v2_0819_15-38-25_test'
env = gym.make('Acrobot-v2')
SEED = 3
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)
#if __name__ == '__main__' and env.ser.isOpen():
if __name__ == '__main__':
    observation_n = env.observation_space.shape
    action_n = env.action_space.n
    hidden_n = 128

    model = a2c_model(observation_n, hidden_n, action_n, load_dir=_load_dir)
    agent = a2c_agent(model, lr=1e-3, sampling_time=0.08, suffix="_test")
    agent.init_message(INIT_MESSAGE)
    file_backup(agent.log_dir)

    agent.train(env)