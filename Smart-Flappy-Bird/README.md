# Smart-Flappy-Bird
Implementation of Smart Flappy Bird where the bird learns to fly on its own using Reinforcement Learning in PyTorch

# Reinforement Learning
[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is an area of machine learning concerned with how 
software agents ought to take actions in an environment so as to maximize some notion of cumulative reward.

# Q-Learning
Q-learning is a reinforcement learning technique used in machine learning. The goal of Q-Learning is to learn a policy, which tells an 
agent what action to take under what circumstances. It does not require a model of the environment and can handle problems with stochastic 
transitions and rewards, without requiring adaptations.

For any finite Markov decision process (FMDP), Q-learning finds a policy that is optimal in the sense that it maximizes the expected value 
of the total reward over all successive steps, starting from the current state.Q-learning can identify an optimal action-selection policy 
for any given FMDP, given infinite exploration time and a partly-random policy."Q" names the function that returns the reward used to 
provide the reinforcement and can be said to stand for the "quality" of an action taken in a given state.
  
# Deep Q-Learning
The DeepMind system used a deep convolutional neural network, with layers of tiled convolutional filters to mimic the effects of 
receptive fields. Reinforcement learning is unstable or divergent when a nonlinear function approximator such as a neural network is 
used to represent Q. This instability comes from the correlations present in the sequence of observations, the fact that small 
updates to Q may significantly change the policy and the data distribution, and the correlations between Q and the target values.

The technique used experience replay, a biologically inspired mechanism that uses a random sample of prior actions instead of the 
most recent action to proceed.This removes correlations in the observation sequence and smooths changes in the data distribution. 
Iterative update adjusts Q towards target values that are only periodically updated, further reducing correlations with the target.

# Requirements

1. PyTorch
2. Python
3. OpenCV
4. PyGame
5. Numpy

# Usage

Training

```python flappy_bird_deep_Q_network.py train```

Testing

```python flappy_bird_deep_Q_network.py test```

*While training, fps was set to maximum frames possible*

*While testing make sure to set the fps to 30 for smooth experience*

```FPS can be changed in game/flappy_bird.py ```
