import sys, os, shutil
from time import sleep
from A2C_AGENT import a2c_agent
from A2C_MODEL import a2c_model
import gym

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
_load_dir = sys.argv[1] if len(sys.argv) > 1 else ""
new_start = bool(sys.argv[2]) if len(sys.argv) > 2 else False

'''
env can be gym environment or serial environment
you should change some codes in A2C_AGENT.py when use gym environment
so if you want to use gym environment you should operate codes from simulator directory
'''
env = gym.make('Acrobot-v2')
SEED = 3
env.seed(SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

# when serial is not opened, error message will be printed
if __name__ == '__main__':
    observation_n = env.observation_space.shape
    hidden_n = 128
    action_n = env.action_space.n

    model = a2c_model(observation_n, hidden_n, action_n, load_dir=_load_dir)
    agent = a2c_agent(model, lr="12e-5", sampling_time=0.04, version='v5', suffix="r1", nstart=new_start)
    agent.init_message(INIT_MESSAGE)
    file_backup(agent.log_dir)

    agent.train(env)

    
