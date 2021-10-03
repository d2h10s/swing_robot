import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os, glob

class a2c_model(tf.keras.Model):
    def __init__(self, observation_n, hidden_n, action_n, load_dir=""):
        super().__init__()
        self.observation_n = observation_n
        self.hidden_n = hidden_n
        self.action_n = action_n
        self.load_dir = load_dir
        print(self.observation_n)
        if not self.load_dir:
            self.input_layer  = layers.Input(shape=self.observation_n, dtype='float32')
            self.fc1_layer    = layers.Dense(self.hidden_n, activation='relu', kernel_initializer=keras.initializers.HeNormal(seed=41), name='Dense1')(self.input_layer)
            self.fc2_layer    = layers.Dense(self.hidden_n, activation='relu', kernel_initializer=keras.initializers.HeNormal(seed=41), name='Dense2')(self.fc1_layer)
            self.actor_layer  = layers.Dense(self.action_n, name='Actor')(self.fc2_layer)
            self.critic_layer = layers.Dense(1, name='Critic')(self.fc2_layer)
            self.nn = keras.Model(inputs=self.input_layer, outputs=[self.actor_layer, self.critic_layer])
        else:
            print('Model is loaded from '+load_dir)
            self.load_dir = os.path.join(os.getcwd(), 'logs',load_dir)
            self.model_dir = os.path.join(self.load_dir, 'tf_model')
            self.max_dir = max(glob.glob(os.path.join(self.model_dir, '**')))
            self.model_dir = os.path.join(self.model_dir, self.max_dir)
            self.nn = keras.models.load_model(self.model_dir)
            print(f'model loaded from {self.model_dir}')
        print(self.nn.summary())
    
    def call(self, x):
        if not self.load_dir:
            x = tf.expand_dims(x, axis=0)
        x = tf.convert_to_tensor(x)
        return self.nn(x)