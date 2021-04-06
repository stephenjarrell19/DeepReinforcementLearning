from network_modules import *

class Agent():
    
    def __init__(self, env_name, lr_actor=0.001, lr_critic=0.002, env=None, gamma=0.95,
                buffer_size = 10**6, rho = 0.001, layer_dims =(400,300), batch_size=64,
                b_normalize = True, state_size = None, action_size = None):
        self.save_dir = env_name + '_models'
        
        if action_size == None:
            self.num_actions = env.action_space.shape[0]
        else:
            self.num_actions = action_size
        
        if state_size == None:        
            self.state_size = env.observation_space.shape[0]
        else:
            self.state_size = state_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.env = env
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.rho = rho
        self.d1_dims = layer_dims[0]
        self.d2_dims = layer_dims[1]
        self.batch_size = batch_size
        self.noise = OUActionNoise(mu=np.zeros(self.num_actions))
        self.action_range = (env.action_space.high[0], env.action_space.low[0])
        self.b_normalize = b_normalize
        self.action_counter = 0
        
        ## Define Replay Buffer
        self.buffer_size = buffer_size
        self.memory = ExperienceReplay(self.buffer_size, self.state_size, self.num_actions)
        
        ## Define Neural Networks
        self.actor = Actor(save_dir = self.save_dir, num_actions = self.num_actions, 
                           d1_dims=self.d1_dims, d2_dims=self.d2_dims, 
                           action_range = self.action_range, b_normalize = self.b_normalize)
        self.target_actor = Actor(save_dir = self.save_dir, num_actions = self.num_actions, 
                                  d1_dims=self.d1_dims, d2_dims=self.d2_dims, 
                                  action_range = self.action_range, 
                                  name = 'target_actor', b_normalize = self.b_normalize)
        
        self.critic = Critic(actions = self.num_actions, save_dir = self.save_dir, d1_dims=self.d1_dims,
                             d2_dims=self.d2_dims, b_normalize = self.b_normalize)
        self.target_critic = Critic(actions = self.num_actions,d1_dims=self.d1_dims, d2_dims=self.d2_dims, 
                                    name = 'target_critic', save_dir = self.save_dir, 
                                    b_normalize = self.b_normalize)
        
        ## Compile the Networks
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_actor))
        self.target_actor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_actor))
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_critic))
        self.target_critic.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_critic))
        
        ## Hard Copy Weights to Target Network
        self.soft_update_weights(rho=1)
        
    def soft_update_weights(self, rho = None):
        """
        Use polyak averaging as a soft weight update to the on-policy network
        
        https://spinningup.openai.com/en/latest/algorithms/ddpg.html
        """
        
        if rho == None:
            rho = self.rho
        
        ## Update Actor
        weights = []
        target_weights = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight*rho + target_weights[i]*(1-rho))
        self.target_actor.set_weights(weights)
        
        ## Update Critic
        weights = []
        target_weights = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*rho + target_weights[i]*(1-rho))
        self.target_critic.set_weights(weights)
        
    def remember_experience(self, state, action, reward, state_p, terminal):
        self.memory.memorize(state, action, reward, state_p, terminal)
    
    
    def compute_action(self, state, epsilon = 0, evaluate=True):
        
        if evaluate:
            action = self.actor(tf.convert_to_tensor([state], dtype='float32'))
        
        ## Epsilon greedy is just for experimentation: not a practical exploration strategy
        
        elif epsilon != 0:
            if np.random.random() > epsilon:
                action = self.actor(tf.convert_to_tensor([state], dtype='float32'))
                action += self.noise()
            else:
                return self.env.action_space.sample()
        else:
            
            ## Add OU temporally correlated noise to the action vector, a very effective exploration strategy
            action = self.actor(tf.convert_to_tensor([state], dtype='float32'))
            action += self.noise()
        
        action = np.clip(action, a_min = min(self.action_range), 
                         a_max = max(self.action_range))
        
        return action[0]
        
    def save_weights(self, debug = True, iteration = None):
        
        if iteration == None:
            iteration = self.actor.save_dir
            
        if debug:
            print(f"\n........Initializing save at Episode {iteration}........")
            print("Saving Actor, Critic, and Target Networks..................")

        if os.path.isdir(self.save_dir) == False:
            os.mkdir(self.save_dir)
        self.actor.save_weights(self.actor.file_checkpoint(self.save_dir))
        self.critic.save_weights(self.critic.file_checkpoint(self.save_dir))
        self.target_actor.save_weights(self.target_actor.file_checkpoint(self.save_dir))
        self.target_critic.save_weights(self.target_critic.file_checkpoint(self.save_dir))
    
    def load_weights(self):
        
        print(f"........Loading Weights........")
        print("Loading Actor.....................")
        self.actor.load_weights(self.actor.file_checkpoint(self.save_dir))
        
        print("Loading Critic.....................")
        self.critic.load_weights(self.critic.file_checkpoint(self.save_dir))
    
        
        print("Loading Target Networks.............")
        self.target_actor.load_weights(self.target_actor.file_checkpoint(self.save_dir))
        self.target_critic.load_weights(self.target_critic.file_checkpoint(self.save_dir))
        print("Load Complete.")
        
    def learn(self):
        
        if self.memory.memory_index < self.batch_size:
            return
        
        ## Sample a Batch from Memory
        states, actions, rewards, states_p, terminals = self.memory.sample_memory(self.batch_size)
        
        with tf.GradientTape() as tape:
            
            ## Feed s' to target_actor
            mu_p = self.target_actor(states_p)
            
            # Feed s' and target_actor output to target_critic
            q_p = tf.squeeze(self.target_critic(states_p, mu_p),1)
            
            ## Target is generated by target_critic, reward if terminal
            target = rewards + self.gamma*q_p*(1-terminals)
            
            # Feed states and actions to crtic
            q = tf.squeeze(self.critic(states,actions), 1)
            
            ## Loss is the TD Error
            critic_loss = tf.keras.losses.MSE(q, target)
            
        ## Calculate the gradient of the loss w.r.t. the critic parameters
        critic_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradient, self.critic.trainable_variables))
        
        with tf.GradientTape() as tape:
            
            ## Actions of actor based on current weights
            actor_actions = self.actor(states)
            
            ## Gradient ASCENT to MAXIMIZE Expected Value over time
            actor_loss = tf.math.reduce_mean((-self.critic(states, actor_actions)))
        
        #
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient,self.actor.trainable_variables))
        
        self.soft_update_weights()

    
def train(agent, env, episodes = 200, epsilon = 0, debug = True, save = True, 
          load_checkpoint = False, evaluate = False):
    
    score_history = []
    best_score = env.reward_range[0]
    
    if load_checkpoint:
        n_steps = 0
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember_experience(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_weights()
      


    ## Game Loop
 
    for i in range(episodes):
        state = env.reset()
        terminal = False
        score = 0
        action_sequence = []

        while not terminal:
            
            if evaluate:
                env.render()
            
            action = agent.compute_action(state, epsilon, evaluate = evaluate)
            action_sequence.append(action)

            state_p, reward, terminal, info = env.step(action)

            score += reward

            agent.remember_experience(state, action, reward, state_p, terminal)

            if not evaluate:
                agent.learn()

            state = state_p


        score_history.append(score)
        avg_score = np.mean(score_history[-75:])

        if avg_score > best_score:
            best_score = avg_score
            if save:
                agent.save_weights(debug=debug, iteration=i)

        if debug:
            print('Ep: ', i, 'Score %.1f' % score, 'Avg %.1f' % avg_score, 
                  'Actions:', len(action_sequence))
        elif i % 10 == 0:
            print('Ep: ', i, 'Score %.1f' % score, 'Avg %.1f' % avg_score, 
                      'Actions:', len(action_sequence))
    if evaluate:
        env.close()
        
    return score_history

