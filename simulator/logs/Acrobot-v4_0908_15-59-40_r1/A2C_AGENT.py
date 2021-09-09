import io, os, yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from pytz import timezone, utc
from datetime import datetime as dt
from typing import Tuple, List
from functools import reduce

STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

tf.config.set_visible_devices([], 'GPU')
class a2c_agent():
    def __init__(self, model, lr=1e-3, sampling_time=0.025, version="", suffix=""):
        self.model = model

        if not model.load_dir:
            self.GAMMA = .99
            self.MAX_STEP = 1000
            self.ALPHA = 0.01
            self.LEARNING_RATE = lr
            self.EPSILON = 1e-3
            self.MAX_DONE = 20
            self.EPS = np.finfo(np.float32).eps.item()

            self.SUFFIX = suffix
            self.sampling_time = sampling_time

            self.done_cnt = 0
            self.num_episode = 1
            self.episode_reward = 0
            self.EMA_reward = 0
            self.loss = 0

            self.most_freq = 0
            self.sigma = 0
            self.plot_img = ''

            self.start_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
            self.start_time_str = dt.strftime(self.start_time, '%m%d_%H-%M-%S')
            self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', f'Acrobot-{version}_{self.start_time_str}_{self.SUFFIX}')
            os.mkdir(self.log_dir)
            os.mkdir(os.path.join(self.log_dir, 'fft_img'))

        else:
            self.log_dir = model.load_dir
            with open(os.path.join(self.log_dir, 'backup.yaml')) as f:
                yaml_data = yaml.safe_load(f)
                self.start_time_str = yaml_data['START_TIME']
                self.start_time = dt.strptime('2021_'+self.start_time_str, '%Y_%m%d_%H-%M-%S')
                self.GAMMA = float(yaml_data['GAMMA'])
                self.MAX_STEP = int(yaml_data['MAX_STEP'])
                self.ALPHA = float(yaml_data['ALPHA'])
                self.LEARNING_RATE = float(yaml_data['LEARNING_RATE'])
                self.EPSILON = float(yaml_data['EPSILON'])
                self.MAX_DONE = float(yaml_data['MAX_DONE'])

                self.num_episode = int(yaml_data['EPISODE'])+1
                self.episode_reward = float(yaml_data['EPISODE_REWARD'])
                self.EMA_reward = float(yaml_data['EMA_REWARD'])
                self.SUFFIX = yaml_data['SUFFIX']
                self.sampling_time = yaml_data['SAMPLING_TIME']

        self.summary_writer = tf.summary.create_file_writer(self.log_dir)


        self.optimizer = keras.optimizers.Adam(learning_rate=self.LEARNING_RATE, epsilon=self.EPSILON)
        self.huber_loss = keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        

    def init_message(self, msg):
        with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
            f.write(msg+'\n\n')


    def fft(self, deg_list: list):
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
        most_freq = freq[np.argmax(fft_mag_data)]
        x = np.arange(n)*self.sampling_time
        sigma = np.max(fft_mag_data)/np.mean(fft_mag_data)
        plt.figure(figsize=(15,10))
        plt.subplot(2,1,1)
        plt.title('FFT')
        plt.plot(x, deg_list)
        plt.xlabel('sec')
        plt.ylabel('deg')
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.grid(True)
        plt.ylabel('mag')
        plt.xlabel('frequency')
        plt.plot(freq, fft_mag_data, linestyle=' ', marker='^', linewidth=1)
        plt.vlines(freq, [0], fft_mag_data)
        plt.xlim([0, 4])
        plt.legend([f'most freq:{most_freq:2.3f}Hz', f'sigma: {sigma:5.2f}'])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        if self.num_episode % 100 == 0 or self.num_episode == 1:
            plt.savefig(os.path.join(self.log_dir, 'fft_img', f'fft{self.num_episode}.png'))
        plt.close()
        buf.seek(0)
        plot_image = tf.image.decode_png(buf.getvalue(), channels=4)
        plot_image = tf.expand_dims(plot_image, 0)

        return most_freq, sigma, plot_image

    
    def get_expected_return(self, rewards: list, standardize: bool = True) -> list:
        """Compute expected returns per timestep."""
        returns = []

        # Start from the end of `rewards` and accumulate reward sums
        # into the `returns` array
        discounted_sum = .0
        for reward in reversed(rewards):
            discounted_sum = reward + self.GAMMA * discounted_sum
            returns.append(discounted_sum)
        returns.reverse()

        if standardize:
            returns = ((returns - tf.math.reduce_mean(returns)) / (np.std(returns) + self.EPS))

        return returns

    def compute_loss(self, action_probs: list, values: list, returns: list) -> tf.Tensor:
        """Computes the combined actor-critic loss."""

        advantage = np.array(returns, dtype=np.float32) - np.array(values, dtype=np.float32)

        action_log_probs = tf.math.log(action_probs)
        actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

        critic_loss = self.huber_loss(values, returns)
        
        del advantage, action_log_probs

        return actor_loss + critic_loss


    def run_episode(self, initial_state: np.array):
        state = initial_state

        action_probs = [0]*self.MAX_STEP
        values = [0]*self.MAX_STEP
        rewards = [0]*self.MAX_STEP
        degrees = [0]*self.MAX_STEP
        self.action_cnt = [0,0]

        for step in range(self.MAX_STEP):
            action_probs_t, value = self.model(state)
            if not step:
                print(f'start with action_probs = {action_probs_t.numpy()[0]}, value = {value.numpy()[0]}')
            if any(tf.math.is_nan([*action_probs_t[0], *value[0]])):
                raise Exception('Nan value is included in model output')
            action = tf.random.categorical(action_probs_t, 1)[0, 0]
            action_probs[step] = action_probs_t[0, action]
            values[step] = value[0,0]

            state, _, _, _ = self.env.step(action)
            c1, s1, c2, s2, w1, w2 = state
            #reward = 1/np.abs(state[0]+0.1)-1/(1+0.1)
            reward = np.abs(s1) # sin(theta1)
            rewards[step] = reward

            deg = np.rad2deg(np.arctan2(s1, c1))
            degrees[step] = deg

            self.action_cnt[action] += 1

        self.most_freq, self.sigma, self.plot_img = self.fft(degrees)

        del degrees

        return action_probs, values, rewards


    def train_step(self, initial_state: np.array):
        """Runs a model training step."""

        with tf.GradientTape() as tape:
            action_probs, values, rewards = self.run_episode(initial_state)

            returns = self.get_expected_return(rewards, standardize=True)
            self.loss = self.compute_loss(action_probs, values, returns)
        grads = tape.gradient(self.loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        self.episode_reward = np.sum(rewards)

        del action_probs, values, rewards, returns, grads



    def train(self, env):
        self.env = env
        #while env.ser.isOpen():
        while True:
            initial_state = env.reset()
            self.train_step(initial_state)
            self.EMA_reward = self.episode_reward*self.ALPHA + self.EMA_reward * (1-self.ALPHA)
            
            self.write_logs()

            self.done_cnt = self.done_cnt + 1 if 100 < self.sigma < 200 and 0.3 < self.most_freq < 0.55 else 0

            if self.done_cnt > self.MAX_DONE:
                print(f"Solved at episode {self.num_episode} with EMA reward {self.EMA_reward}")
                with self.summary_writer.as_default():
                    tf.summary.image(f'fft of final episode{self.num_episode:05}', self.plot_img, step=0)
                break
            self.num_episode += 1


    def run_test(self, env):
        state = env.reset()
        for step in range(1, self.MAX_STEP):

            action_probs, _ = self.model(state)

            action = np.argmax(action_probs)
            state, *_ = env.step(action)

            with self.summary_writer.as_default():
                tf.summary.scalar('test angle of link1', env.th1, step=step)
    
    
    def write_logs(self):
        now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
        log_text = f"EMA reward: {self.EMA_reward:9.2f} at episode {self.num_episode:5} --freq:{self.most_freq:7.3f} --sigma:{self.sigma:7.2f} --action:({self.action_cnt[0]:4d},{self.action_cnt[1]:4d})--time:{now_time_str}"
        print(log_text)

        with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
            f.write(log_text+'\n')

        with self.summary_writer.as_default():
            tf.summary.scalar('losses', self.loss, step=self.num_episode)
            tf.summary.scalar('reward of episodes', self.episode_reward, step=self.num_episode)
            tf.summary.scalar('frequency of episodes', self.most_freq, step=self.num_episode)
            tf.summary.scalar('sigma of episodes', self.sigma, step=self.num_episode)

        with open(os.path.join(self.log_dir, 'episode-reward-loss-freq-sigma.txt'), 'a') as f:
            f.write(f'{self.num_episode} {self.episode_reward} {self.loss} {self.most_freq} {self.sigma}\n')

        if self.num_episode % 100 == 0 or self.num_episode == 1:
            self.yaml_backup()
            self.model.save(os.path.join(self.log_dir, 'tf_model', f'learning_model{self.num_episode}'))
            with self.summary_writer.as_default():
                tf.summary.image(f'fft of episode{self.num_episode:05}', self.plot_img, step=0)


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
                        'EPISODE_REWARD':   float(self.episode_reward),\
                        'SAMPLING_TIME':    self.sampling_time,\
                        'MAX_DONE':         self.MAX_DONE,\
                        'SUFFIX':           self.SUFFIX}
            yaml.dump(yaml_data, f)