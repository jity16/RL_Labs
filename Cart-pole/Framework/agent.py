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
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

def update_model(opt, memory, dqn, optimizer):
    if len(memory) < opt.batch_size:
        return
    transitions = memory.sample(opt.batch_size)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=opt.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    action_batch = torch.tensor(batch.action)
    # reward_batch = torch.cat(batch.reward)
    reward_batch = torch.tensor(batch.reward)

    # print(dqn(state_batch).shape)
    # exit()
    # print(action_batch.shape)
    # exit()
    state_action_values = dqn(state_batch).gather(0,action_batch)
    next_state_values = torch.zeros(opt.batch_size, device=opt.device)
    next_state_values[non_final_mask] = dqn(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * opt.gamma) + reward_batch
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

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
            update_model(opt, replay_memory, dqn, optimizer)

            if done:
                env.reset()
                print("iteration %d :  score = %d "%(iter, step))
                break
              



        

    





