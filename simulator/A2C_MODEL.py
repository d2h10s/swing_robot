import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple
import os, glob

class a2c_model(tf.keras.Model):
    def __init__(self, observation_n, hidden_n, action_n, load_dir=""):
        super().__init__()
        self.observation_n = observation_n
        self.hidden_n = hidden_n
        self.action_n = action_n
        self.load_dir = load_dir

        if not self.load_dir:
            self.input_layer  = layers.Input(shape=self.observation_n)
            self.fc1_layer    = layers.Dense(self.hidden_n, activation='relu', name='Dense1')(self.input_layer)
            self.fc2_layer    = layers.Dense(self.hidden_n, activation='relu', name='Dense2')(self.fc1_layer)
            self.actor_layer  = layers.Dense(self.action_n, name='Actor')(self.fc2_layer)
            self.critic_layer = layers.Dense(1, name='Critic')(self.fc2_layer)
            self.nn = keras.Model(inputs=self.input_layer, outputs=[self.actor_layer, self.critic_layer])
        else:
            self.load_dir = os.path.join(os.getcwd(), 'logs',load_dir)
            self.model_dir = os.path.join(self.load_dir, 'tf_model')
            print(glob.glob(os.path.join(self.model_dir, '**')))
            self.max_dir = max(glob.glob(os.path.join(self.model_dir, '**')))
            self.model_dir = os.path.join(self.model_dir, self.max_dir)
            self.nn = tf.keras.models.load_model(self.model_dir)
        print(self.nn.summary())
    
    def call(self, state: tf.Tensor)->Tuple[tf.Tensor, tf.Tensor]:
        x = tf.expand_dims(state, axis=0)
        return self.nn(x)