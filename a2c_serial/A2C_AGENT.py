import io, os, yaml, time, cv2, gc, psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from pytz import timezone, utc
from datetime import datetime as dt

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

class a2c_agent():
    '''
    @param model environment which is from gym or serial
    @param lr learning_rate of optimizer
    @param sampling_time period reading observation data from robot
    @param version environment version which is made by user
    @param suffix user's comment
    @param nstart Whether rename folder name and time variables
    '''
    def __init__(self, model, lr='1e-3', sampling_time=0.025, version="", suffix="", nstart=False, test=False):
        self.model = model
        self.EPS = np.finfo(np.float32).eps.item()
        # GAMMA is discount factor
        self.GAMMA = .99
        self.MAX_STEP = 1000
        self.LEARNING_RATE = float(lr)
        self.EPSILON = 1e-3
        # if If successive successes beyond MAX_DONE stop traing because it means training is completed
        self.MAX_DONE = 20
        # gradient cut by number of self.NORM
        self.NORM = 0.5

        self.num_episode = 1
        self.episode_reward = 0
        # EMA_reward means exponential moving everage reward
        self.EMA_reward = 0
        self.ALPHA = 0.01
        self.SUFFIX = f'{suffix}_{lr}'
        self.max_angle = 0
        self.sampling_time = sampling_time

        # m is minimum of observation parameters, and M means maximum
        self.M = [ 1.2165426422741104,   1.7601296440512415, 5.567071950337454,  20.839231268812295]
        self.m = [-1.1972720082172352, -0.19505799720288627, -5.73218793410446, -19.163715186897736]

        matplotlib.use('Agg')
        if test:
            return

        if not model.load_dir or nstart:
            self.start_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
            self.start_time_str = dt.strftime(self.start_time, '%m%d_%H-%M-%S')
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', f'Acrobot-{version}_{self.start_time_str}_{self.SUFFIX}')

        if not model.load_dir:
            os.mkdir(self.log_dir)
            os.mkdir(os.path.join(self.log_dir, 'fft_img'))
            os.mkdir(os.path.join(self.log_dir, 'tf_model'))
            os.mkdir(os.path.join(self.log_dir, 'video'))
            with open(os.path.join(self.log_dir, 'learning_data.txt'), 'a') as f:
                f.write('episode,reward,loss,frequency,sigma\r\n')
        
        # if start from exist model, load parameters from exist log
        else:
            if nstart:
                os.rename(self.model.load_dir, self.log_dir)
                self.model.load_dir = self.log_dir
            print('agent parameter loaded from previous model!')
            self.log_dir = model.load_dir
            with open(os.path.join(self.log_dir, 'backup.yaml')) as f:
                yaml_data = yaml.safe_load(f)
                if not nstart:
                    self.start_time_str = yaml_data['START_TIME']
                    self.start_time = dt.strptime('2021_'+self.start_time_str, '%Y_%m%d_%H-%M-%S')
                    self.start_time.replace(tzinfo=timezone('Asia/Seoul'))
                self.GAMMA = float(yaml_data['GAMMA'])
                self.MAX_STEP = int(yaml_data['MAX_STEP'])
                self.ALPHA = float(yaml_data['ALPHA'])
                self.LEARNING_RATE = float(yaml_data['LEARNING_RATE'])
                self.EPSILON = float(yaml_data['EPSILON'])
                self.MAX_DONE = float(yaml_data['MAX_DONE'])

                self.num_episode = int(yaml_data['EPISODE'])+1
                self.EMA_reward = float(yaml_data['EMA_REWARD'])
                self.SUFFIX = yaml_data['SUFFIX']
                self.sampling_time = yaml_data['SAMPLING_TIME']
        self.summary_writer = tf.summary.create_file_writer(self.log_dir)


        self.optimizer = optimizers.Adam(learning_rate=self.LEARNING_RATE, epsilon=self.EPSILON)
        self.huber_loss = keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        

    def init_message(self, msg):
        with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
            f.write(msg+'\n\n')


    def fft(self, deg_list, act_list):
        Fs = 1/self.sampling_time
        n = len(deg_list)
        scale = n//2
        k = np.arange(n)
        T = n/Fs
        freq = k/T
        freq = freq[range(scale)]
        fft_data = np.fft.fft(deg_list)/n
        fft_data = fft_data[range(scale)]
        fft_mag_data = np.abs(fft_data)
        self.most_freq = freq[np.argmax(fft_mag_data)]
        self.sigma = np.max(fft_mag_data)/np.mean(fft_mag_data)

        x = np.arange(n)*self.sampling_time
        fig = plt.figure(figsize=(15,15))
        ax1 = plt.subplot(3,1,1)
        ax1.plot(x, deg_list)
        ax1.set_title(f'FFT{self.num_episode}')
        ax1.set_xlabel('sec')
        ax1.set_ylabel('deg')
        ax1.grid(True)

        ax2 = fig.add_subplot(3,1,2)
        ax2.grid(True)
        ax2.plot(freq, fft_mag_data, linestyle=' ', marker='^', linewidth=1)
        ax2.vlines(freq, [0], fft_mag_data)
        ax2.set_xlim([0, 4])
        ax2.set_title('Magnitude')
        ax2.set_xlabel('frequency')
        ax2.set_ylabel('mag')
        ax2.legend([f'most freq:{self.most_freq:2.3f}Hz', f'sigma: {self.sigma:5.2f}'])

        ax3 = fig.add_subplot(3,1,3)
        ax3.grid(True)
        ax3.plot(range(1,self.MAX_STEP+1), act_list)
        ax3.set_title('Action')
        ax3.set_xlabel('step')
        ax3.set_ylabel('action')
        ax3.legend([f'action0:{self.MAX_STEP - sum(self.action_cnt)}'])

        #buf = io.BytesIO()
        #fig.savefig(buf, format='png')
        fig.savefig(os.path.join(self.log_dir, 'fft_img', f'fft{self.num_episode}.png'))
        #buf.seek(0)
        plt.clf(); plt.close('all')
        del fig, ax1, ax2, ax3
        gc.collect()
        #plot_image = tf.image.decode_png(buf.getvalue(), channels=4)
        #plot_image = tf.expand_dims(plot_image, 0)


    def get_expected_return(self, rewards: tf.Tensor, standardize: bool = True) -> tf.Tensor:
        """Compute expected returns per timestep."""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant(0.0)
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + self.GAMMA * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / 
                    (tf.math.reduce_std(returns) + self.EPS))

        return returns


    def compute_loss(self, action_probs: tf.Tensor, values: tf.Tensor, returns: tf.Tensor) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = returns - tf.stop_gradient(values) # 역전파 방지

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)
    
        critic_loss = self.huber_loss(values, returns) # for gradient clipping
        print('\ncritic loss is', critic_loss.numpy(), 'actor loss is', actor_loss.numpy(),end=' ')
        del advantage, action_log_probs

        return actor_loss + critic_loss


    def run_episode(self, initial_state: np.array):
        state = initial_state

        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        degrees = np.zeros(self.MAX_STEP, dtype=np.float32)
        self.action_cnt = np.zeros(self.MAX_STEP, dtype=np.int)
        self.max_angle = 0

        if self.num_episode % 100 == 0:
            blue_color = (255, 0, 0) # BGR
            font = cv2.FONT_HERSHEY_SIMPLEX
            video_dir = os.path.join(self.log_dir, 'video', f'{self.num_episode}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            img_shape = self.env.render('rgb_array').shape[:2]
            videoWriter = cv2.VideoWriter(video_dir, fourcc, 15, img_shape)

        for step in range(1, self.MAX_STEP+1):
            start_time = time.time()
            state = np.array([(state[i]-self.m[i])/(self.M[i]-self.m[i]) for i in range(4)])
            action_logits_t, value = self.model(state)
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)
            a0, a1, v = *action_probs_t.numpy()[0], value.numpy()[0,0]
            if any(tf.math.is_nan([a0, a1, v])):
                raise Exception(f'Nan value is included in model output, {a0}, {a1}, {v}')
            self.action_cnt[step-1] = int(action)
            action_probs = action_probs.write(step-1, action_probs_t[0, action])
            values = values.write(step-1, tf.squeeze(value))

            state = self.env.step(action)
            th1, th2, vel1, vel2 = state
            self.max_angle = max(self.max_angle, np.rad2deg(th1))
            
            #reward = -np.abs(np.cos(th1)) # R0
            reward = np.abs(np.sin(th1)) # R1
            #reward = 1/np.abs(np.cos(th1)+0.1)-1/(1+0.1) # R2
            #reward = -np.cos(th1*2) # R3

            rewards = rewards.write(step-1, reward)

            print(f'\r--step {step:5d}  --reward {reward:8.02} --action {action} --action_probs [{a0:8.02} {a1:8.02}] --value [{v:8.02}]', end='')

            deg = np.rad2deg(th1)
            degrees[step-1] = deg
            
            didWait = False
            while time.time() - start_time < self.sampling_time:
                didWait = True
            if not didWait:
                print(f"\rwait time over {int((time.time()-start_time)*1000)}ms at step {step:<70}")
            
            if self.num_episode % 100 == 0:
                img = self.env.render(mode='rgb_array').astype(np.float32)
                cv2.putText(img=img,text=f'TEST: Step({step:04})', org=(50,50), fontFace=font, fontScale=1,color=blue_color, thickness=1, lineType=0)
                cv2.imshow('Actor-Critic', img)
                videoWriter.write(img.astype(np.ubyte))
        if self.num_episode % 100 == 0:
            videoWriter.release()

        self.fft(degrees, self.action_cnt)

        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        return action_probs, values, rewards
    

    def train_step(self, initial_state: np.array):
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.run_episode(initial_state)

            returns = self.get_expected_return(rewards, standardize=True)
            
            self.loss = self.compute_loss(action_probs, values, returns)
            print('loss is', self.loss.numpy())
        grads = tape.gradient(self.loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, self.NORM) for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        self.episode_reward = np.sum(rewards)

        del action_probs, values, rewards, returns, grads


    def train(self, env):
        self.env = env
        self.done_cnt = 0
        self.env.set_zero_angle()
        while self.env.ser.isOpen():
            initial_state = self.env.reset(self.num_episode)
            self.train_step(initial_state)
            if self.num_episode == 1:
                self.EMA_reward = self.episode_reward
            else:
                self.EMA_reward = self.episode_reward*self.ALPHA + self.EMA_reward * (1-self.ALPHA)
            
            self.write_logs()

            self.done_cnt = self.done_cnt + 1 if 100 < self.sigma < 200 and 0.3 < self.most_freq < 0.55 else 0

            if self.done_cnt > self.MAX_DONE:
                print(f"Solved at episode {self.num_episode} with EMA reward {self.EMA_reward}")
                break

            self.num_episode += 1
        cv2.destroyAllWindows()


    def run_test(self, env):
        step = 0
        state = env.reset(1)
        while True:
            
            start_time = time.time()

            action_logits_t, value = self.model(state)

            state = np.array([(state[i]-self.m[i])/(self.M[i]-self.m[i]) for i in range(4)])
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            state = env.step(action)

            didWait = False
            while time.time() - start_time < self.sampling_time:
                didWait = True
            if not didWait:
                print(f"\rwait time over {int((time.time()-start_time)*1000)}ms at step {step:<70}")
            step += 1


    def write_logs(self):
        now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
        a1 = sum(self.action_cnt)
        a0 = self.MAX_STEP - a1
        log_text = f"reward: {self.episode_reward:9.2g} --episode: {self.num_episode:5} --max angle:{self.max_angle:5.2f} --freq:{self.most_freq:7.3f} --sigma:{self.sigma:7.2f} --action:({a0:4d},{a1:4d})--time:{now_time_str}"
        print(log_text)

        with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
            f.write(log_text+'\n')

        cpu, mem = self._check_usage_of_cpu_and_memory()

        with self.summary_writer.as_default():
            tf.summary.scalar('action1 ratio', sum(self.action_cnt)/self.MAX_STEP, step=self.num_episode)
            tf.summary.scalar('losses', self.loss, step=self.num_episode)
            tf.summary.scalar('reward of episodes', self.episode_reward, step=self.num_episode)
            tf.summary.scalar('frequency of episodes', self.most_freq, step=self.num_episode)
            tf.summary.scalar('sigma of episodes', self.sigma, step=self.num_episode)
            tf.summary.scalar('max angle of episodes', self.max_angle, step=self.num_episode)
            tf.summary.scalar('memory usage of episodes', mem, step=self.num_episode)

        with open(os.path.join(self.log_dir, 'learning_data.txt'), 'a') as f:
            f.write(f'{self.num_episode},{self.episode_reward},{self.loss},{self.most_freq},{self.sigma}\r\n')

        if self.num_episode % 100 == 0 or self.num_episode == 1:
            self.yaml_backup()
            self.model.save(os.path.join(self.log_dir, 'tf_model', f'learning_model{self.num_episode}'))
            gc.collect()
                

    def yaml_backup(self):
        now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
        with open(os.path.join(self.log_dir, 'backup.yaml'), 'w') as f:
            yaml_data = {'START_TIME':      self.start_time_str,\
                        'ELAPSED_TIME':     str(now_time-self.start_time),\
                        'END_TIME':         now_time_str,\
                        'GAMMA':            self.GAMMA,\
                        'MAX_STEP':         self.MAX_STEP,\
                        'ALPHA':            self.ALPHA,\
                        'LEARNING_RATE':    self.LEARNING_RATE,\
                        'EPSILON':          self.EPSILON,\
                        'EPISODE':          self.num_episode,\
                        'EMA_REWARD':       float(self.EMA_reward),\
                        'SAMPLING_TIME':    self.sampling_time,\
                        'MAX_DONE':         self.MAX_DONE,\
                        'SUFFIX':           self.SUFFIX}
            yaml.dump(yaml_data, f)

    def _check_usage_of_cpu_and_memory(self):
        pid = os.getpid()
        py  = psutil.Process(pid)
        
        cpu_usage   = os.popen("ps aux | grep " + str(pid) + " | grep -v grep | awk '{print $3}'").read()
        cpu_usage   = cpu_usage.replace("\n","")
        
        memory_usage  = round(py.memory_info()[0] /2.**30, 2)

        return cpu_usage, memory_usage