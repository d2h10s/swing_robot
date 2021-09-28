import io, os, yaml, time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from PIL import Image
from pytz import timezone, utc
from datetime import datetime as dt


STX = b'\x02'
ETX = b'\x03'
ACK = b'\x06'
NAK = b'\x15'

class a2c_agent():
    def __init__(self, model, lr='1e-3', sampling_time=0.025, version="", suffix=""):
        self.model = model
        self.EPS = np.finfo(np.float32).eps.item()
        self.GAMMA = .99
        self.MAX_STEP = 1000
        self.ALPHA = 0.01
        self.LEARNING_RATE = float(lr)
        self.EPSILON = 1e-3
        self.MAX_DONE = 20
        self.NORM = 0.5

        self.num_episode = 1
        self.episode_reward = 0
        self.EMA_reward = 0
        self.SUFFIX = f'{suffix}_{lr}'
        self.sampling_time = sampling_time

        self.start_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        self.start_time_str = dt.strftime(self.start_time, '%m%d_%H-%M-%S')
        self.log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'logs', f'Acrobot-{version}_{self.start_time_str}_{self.SUFFIX}')
        if not model.load_dir:
            os.mkdir(self.log_dir)
            os.mkdir(os.path.join(self.log_dir, 'fft_img'))
            os.mkdir(os.path.join(self.log_dir, 'tf_model'))

        else:
            print('agent parameter loaded from previous model!')
            self.log_dir = model.load_dir
            with open(os.path.join(self.log_dir, 'backup.yaml')) as f:
                yaml_data = yaml.safe_load(f)
                #self.start_time_str = yaml_data['START_TIME']
                #self.start_time = dt.strptime('2021_'+self.start_time_str, '%Y_%m%d_%H-%M-%S')
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
        most_freq = freq[np.argmax(fft_mag_data)]
        x = np.arange(n)*self.sampling_time
        sigma = np.max(fft_mag_data)/np.mean(fft_mag_data)
        plt.figure(figsize=(15,15))
        plt.subplot(3,1,1)
        plt.title('FFT')
        plt.plot(x, deg_list)
        plt.xlabel('sec')
        plt.ylabel('deg')
        plt.grid(True)

        plt.subplot(3,1,2)
        plt.grid(True)
        plt.ylabel('mag')
        plt.xlabel('frequency')
        plt.plot(freq, fft_mag_data, linestyle=' ', marker='^', linewidth=1)
        plt.vlines(freq, [0], fft_mag_data)
        plt.xlim([0, 4])
        plt.legend([f'most freq:{most_freq:2.3f}Hz', f'sigma: {sigma:5.2f}'])

        plt.subplot(3,1,3)
        plt.grid(True)
        plt.title('Action')
        plt.ylabel('action')
        plt.xlabel('step')
        plt.plot(range(1,self.MAX_STEP+1), act_list)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        #if self.num_episode % 100 == 0 or self.num_episode == 1:
        plt.savefig(os.path.join(self.log_dir, 'fft_img', f'fft{self.num_episode}.png'))
        plt.close()
        buf.seek(0)
        plot_image = tf.image.decode_png(buf.getvalue(), channels=4)
        plot_image = tf.expand_dims(plot_image, 0)
        return most_freq, sigma, plot_image


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

        for step in range(1, self.MAX_STEP+1):
            start_time = time.time()
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
            #reward = -np.abs(np.cos(th1))
            reward = np.abs(np.sin(th1))
            #reward = 1/np.abs(np.cos(th1)+0.1)-1/(1+0.1)
            #reward = -np.cos(th1*2)
            rewards = rewards.write(step-1, reward)

            print(f'\r--step {step:5d}  --reward {reward:8.02} --action {action} --action_probs [{a0:8.02} {a1:8.02}] --value [{v:8.02}]', end='')

            deg = np.rad2deg(th1)
            degrees[step-1] = deg
            
            didWait = False
            while time.time() - start_time < self.sampling_time:
                didWait = True
            if not didWait:
                print(f"\rwait time over {int((time.time()-start_time)*1000)}ms at step {step:<70}")
            


        self.most_freq, self.sigma, self.plot_img = self.fft(degrees, self.action_cnt)
        del degrees

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
        done_cnt = 0
        self.env.set_zero_angle()
        while self.env.ser.isOpen():
            initial_state = self.env.reset()
            self.train_step(initial_state)
            if self.num_episode == 1:
                self.EMA_reward = self.episode_reward
            else:
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
            state = env.step(action)
            th1 = np.rad2deg(state[0]) # deg
            with self.summary_writer.as_default():
                tf.summary.scalar('test angle of link1', th1, step=step)


    def write_logs(self):
        now_time = utc.localize(dt.utcnow()).astimezone(timezone('Asia/Seoul'))
        now_time_str = dt.strftime(now_time, '%m-%d_%Hh-%Mm-%Ss')
        a1 = sum(self.action_cnt)
        a0 = self.MAX_STEP - a1
        log_text = f"reward: {self.episode_reward:9.2g} --episode: {self.num_episode:5} --max angle:{self.env.max_angle:5.2f} --freq:{self.most_freq:7.3f} --sigma:{self.sigma:7.2f} --action:({a0:4d},{a1:4d})--time:{now_time_str}"
        print(log_text)

        with open(os.path.join(self.log_dir, 'terminal_log.txt'), 'a') as f:
            f.write(log_text+'\n')

        with self.summary_writer.as_default():
            tf.summary.scalar('losses', self.loss, step=self.num_episode)
            tf.summary.scalar('reward of episodes', self.episode_reward, step=self.num_episode)
            tf.summary.scalar('frequency of episodes', self.most_freq, step=self.num_episode)
            tf.summary.scalar('sigma of episodes', self.sigma, step=self.num_episode)
            tf.summary.scalar('max angle o episodes', self.env.max_angle, step=self.num_episode)

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
                        'SAMPLING_TIME':    self.sampling_time,\
                        'MAX_DONE':         self.MAX_DONE,\
                        'SUFFIX':           self.SUFFIX}
            yaml.dump(yaml_data, f)