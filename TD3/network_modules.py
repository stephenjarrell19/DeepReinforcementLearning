import tensorflow as tf
import os
import gym
import numpy as np
from tensorflow.keras.initializers import RandomNormal

class Actor(tf.keras.Model):
    
    """
    Chooses an action given a state
    
    """
    def __init__(self, num_actions, d1_dims=256, d2_dims=256, action_range = (-1,1),
                 save_dir='model_weights/', name = 'actor', b_normalize = True):
        
        super().__init__()
        self.actions = num_actions
        self.d1_dims = d1_dims
        self.d2_dims = d2_dims
        self.action_range = action_range
        self.batch_normalize = b_normalize
        
        ## Create Model Checkpoint Destination
        self.model_name = name
        self.save_dir = save_dir
        
        
        ## Build the Fully Connected Model Layers
        initializer1 = RandomNormal(0., 3*10**-(self.d1_dims/100))
        initializer2 = RandomNormal(0., 3*10**-(self.d2_dims/100))
        if self.batch_normalize:
            self.batch_norm1 = tf.keras.layers.BatchNormalization(trainable=False, center=False)
            self.batch_norm2 = tf.keras.layers.BatchNormalization(trainable=False, center=False)
     
        self.d1 = tf.keras.layers.Dense(self.d1_dims, kernel_initializer=initializer1,
                                                  activation='relu')
        
        self.d2 = tf.keras.layers.Dense(self.d2_dims, kernel_initializer=initializer2,
                                                  activation='relu')
        self.action_vector = tf.keras.layers.Dense(self.actions, activation = 'tanh') #[-1,1]
    
    def file_checkpoint(self, save_dir = None):
        if save_dir == None:
            save_dir = self.save_dir
        return os.path.join(save_dir, self.model_name +'.h5')
    
    def call(self, state):
        
        
        ## Forward propagation
        if self.batch_normalize:
            mu = self.d1(state)
            mu = self.batch_norm1(mu)
            mu = self.d2(mu)
            mu = self.batch_norm2(mu)
            mu = self.action_vector(mu)
        else:
            mu = self.d1(state)
            mu = self.d2(mu)
            mu = self.action_vector(mu)
            
        ## Multiplay the tanh output by 
        mu = mu * max(self.action_range)
        
        return mu


class Critic(tf.keras.Model):
    
    """
    Evaluates the value by the state and actions from the Actor Network
    
    """
    def __init__(self, actions, d1_dims=256, d2_dims=256, save_dir='model_weights/', 
                 name = 'critic', b_normalize = True):
        
        super().__init__()
        self.actions = actions
        self.d1_dims = d1_dims
        self.d2_dims = d2_dims
        self.batch_normalize = b_normalize
        
        ## Create Model Checkpoint Destination
        self.model_name = name
        self.save_dir = save_dir
        
        
        ## Build the Fully Connected Model Layers
        initializer1 = RandomNormal(0., 3*10**-(self.d1_dims/100))
        initializer2 = RandomNormal(0., 3*10**-(self.d2_dims/100))
        if self.batch_normalize:
            self.batch_norm1 = tf.keras.layers.BatchNormalization(trainable=False)
            self.batch_norm2 = tf.keras.layers.BatchNormalization(trainable=False)
            
        self.d1 = tf.keras.layers.Dense(self.d1_dims, kernel_initializer=initializer1,
                                                  activation='relu')
        self.d2 = tf.keras.layers.Dense(self.d2_dims, kernel_initializer=initializer2,
                                                  activation='relu')
        self.q = tf.keras.layers.Dense(1, activation = None)
    
    def file_checkpoint(self, save_dir = None):
        if save_dir == None:
            save_dir = self.save_dir
        return os.path.join(save_dir, self.model_name +'.h5')
    
    def call(self, state, action):
        
        ## Forward propagation
        if self.batch_normalize:
            q = self.d1(tf.concat([state,action], axis=1))
            q = self.batch_norm1(q)
            q = self.d2(q)
            q = self.batch_norm2(q)
            q = self.q(q)
        else:
            q = self.d1(tf.concat([state,action], axis=1))
            q = self.d2(q)
            q = self.q(q)
            
        return q

class OUActionNoise():
    
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

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