import argparse 
import os
import Network.nn as nn
from Network.er import ReplayMemory
import torch
import gym
import random
import torch
import torch.optim as optim
from itertools import count
import numpy as np

def init_models(opt):
    net = nn.DQN(opt)
    net.apply(nn.weights_init)
    if opt.load_dir != '':
        net.load_state_dict(torch.load(opt.load_dir))
    return net

def select_actions(net, state, epsilon, n_actions):
    if random.random() > epsilon:
        with torch.no_grad():
            return torch.argmax(net(state)[0]).item()
    else:
        return random.randrange(n_actions)

def obv2state(state, observation):
    # TODO: combine several states
    single_state = torch.zeros((1,84,84), dtype = torch.float32)
    single_state[:, 0, 0:4] = torch.from_numpy(observation)
    state = torch.cat((state.squeeze(0)[1:,:,:], single_state)).unsqueeze(0)
    return state



def train(opt):
    env = gym.make(opt.env)
    env.reset()

    # TODO: policy_net and target_net
    opt.labels= env.action_space.n  # nn.fc out
    dqn = init_models(opt)
    optimizer = optim.RMSprop(dqn.parameters())

    replay_memory = ReplayMemory(opt.rm_size)

    epsilon_decrements = np.linspace(opt.eps_start, opt.eps_end, opt.niters)

  
    
    # print(state.shape)
    for iter in range(opt.niters):
          # start
        env.render()
        observation, reward, done, info = env.step(random.randrange(opt.labels))

        # TODO： 合并到obv2state中
        single_state = torch.zeros((1,84,84), dtype = torch.float32)
        single_state[:, 0, 0:4] = torch.from_numpy(observation)
        state = torch.cat((single_state, single_state, single_state, single_state)).unsqueeze(0)

        for step in count():
            action = select_actions(dqn, state, epsilon_decrements[iter], opt.labels)

            observation, reward, done, info = env.step(action)
            if not done:
                next_state = obv2state(state, observation)
            else:
                next_state = None
            # Store the transition in memory
            replay_memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # TODO: update 

            if done:
                env.reset()
                print("iteration %d :  score = %d "%(iter, step))
                break
              



        

    





