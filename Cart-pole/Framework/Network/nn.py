from typing import OrderedDict
import torch
import torch.nn as nn

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)
   
# TODO: Class conv_blocks
# TODO: in_size, out_size auto compute
class DQN(nn.Module):

    def __init__(self, opt):
        super(DQN, self).__init__()
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 16, 8, 4)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.conv2d(16, 32, 4, 2)),
            ('relu2', nn.ReLU(inplace=True))
        ]))
        self.tail = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2592, 256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(256, opt.labels))
        ]))
        
    def forward(self, x):
        x = self.body(x)
        x = self.tail(x)
        return x
