import Network.nn as nn
import torch
import gym

def init_models(opt):
    model = nn.DQN(opt)
    model.apply(nn.weights_init)
    if opt.load_dir != '':
        model.load_state_dict(torch.load(opt.load_dir))
    return model

def train(opt):
    env = gym.make(opt.env)
    # print(env.action_space)
    # print(env.observation_space)
    opt.labels= env.action_space.n  # nn.fc out
    for iter in range(opt.niters):
        env.reset()
        print("hello iter:",iter)
    
    





