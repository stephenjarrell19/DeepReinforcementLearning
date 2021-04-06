import tensorflow as tf
import numpy as np
import os

class ExperienceReplay():
    
    """
    Experience replay stablizes learning. This buffer
    should be large enough to capture a wide range of experiences so that
    it may generalize well.
    
    The buffer saves the (s, a, s', r, d) for each step in an environment.
    The models will randomly sample experiences using a uniform distribution
    later, to update the deep neural networks.
    
    Hyper-parameters:
        memory_capacity - Too large and training is slow. Too small and
            training will overfit to most recent experience
    """
    
    def __init__(self, size, state_dims, num_actions):

        self.memory_capacity = size
        self.memory_index = 0
        
        ## Initialize a memory array for (s, a, s', r, d)
        self.state_memory = np.zeros((self.memory_capacity, state_dims))
        self.action_memory = np.zeros((self.memory_capacity, num_actions))
        self.reward_memory = np.zeros(self.memory_capacity)
        self.state_p_memory = np.zeros((self.memory_capacity, state_dims))
        self.terminal_memory = np.zeros(self.memory_capacity, dtype=np.bool_)
        
            
    def memorize(self, state, action, reward, state_p, terminal):
        
        ## Overwrite buffer when full
        index = self.memory_index % self.memory_capacity
        
        ## Store Experience
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.state_p_memory[index] = state_p
        self.terminal_memory[index] = terminal
        
        self.memory_index += 1
    
    def sample_memory(self, batch_size):
         
        sample_size = min(self.memory_index, self.memory_capacity)
        
        ## Randomly sample a batch of memories without replacement
        batch = np.random.choice(sample_size, batch_size, replace=False)
        
        ## Generate Batch by Indices
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_p = self.state_p_memory[batch]
        terminals = self.terminal_memory[batch]
        
        ## Numpy to Tensor
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_p = tf.convert_to_tensor(states_p, dtype=tf.float32)
        #terminals = tf.convert_to_tensor(terminals, dtype=tf.float32)
        
        return states, actions, rewards, states_p, terminals

class DQN(tf.keras.Model):
    
    def __init__(self, input_dims, actions, c1=32, c2=64, c3=64, d1=256):
        super().__init__()
        self.model = tf.keras.Sequential(
                    [
                        tf.keras.layers.Conv2D(32, kernel_size=8, input_shape=img_shape,
                                               activation='relu', strides=2),
                        tf.keras.layers.Conv2D(64, kernel_size=6,
                                               strides=2, activation='relu'),
                        tf.keras.layers.Conv2D(64, kernel_size=6,
                                               strides=2, activation='relu'),
                        tf.keras.layers.Dense(256, activation='relu'),
                        tf.keras.layers.Dense(actions, activation=None)
                     ])
        model.compile(loss= tf.keras.losses.Huber(), optimizer = 'adam',
                     metrics = ['accuracy'])
        
        
        
        