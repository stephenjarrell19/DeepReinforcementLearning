# Deep Reinforcement Learning: Deep Deterministic Policy Gradient (DDPG)
A Deep Deterministic Policy Gradient implementation from scratch using Tensorflow and Numpy, to solve OpenAI Gym environments with continuous actions and large state spaces. This Deep Reinforcement Learning algorithm is adaptable to any environment formatted for the OpenAI Gym API.

## Running the Code in an Environment
To run the code, open up the ```agent.ipynb``` notebook, which should import the network modules and agent module. The ```Agent()``` class I've defined initializes upon receiving an environment object in the ```env``` argument. The code takes some time to run, so I've included the Pendulum environment as a proof-of-concept to test the model on. The Pendulum environment takes about <100 episodes to converge to an optimal policy. 

### Running the BipedalWalker-v2 Environment
The Bipedal Walker takes many more episodes to run since there are more network parameters, more complex reward functions, a larger state space, and more actions to perform. I would recommend a minimum of 800 episodes and a maximum of 3000. The computational cost is high, especially since episodes take longer as the agent learns not to fall over and terminate the episode early.
