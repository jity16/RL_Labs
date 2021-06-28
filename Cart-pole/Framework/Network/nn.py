from typing import OrderedDict
import torch
import torch.nn as nn

def weights_init(m):
    if type(m) == nn.Conv2d:
        m.weight.data.normal_(0.0, 0.02)
    elif type(m) == nn.Linear:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.01)
   
# TODO: Class conv_blocks
# TODO: in_size, out_size auto compute
class DQN(nn.Module):

    def __init__(self, opt):
        super(DQN, self).__init__()
        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 16, 8, 4)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(16, 32, 4, 2)),
            ('relu2', nn.ReLU(inplace=True))
        ]))
        self.tail = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2592, 256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc2', nn.Linear(256, opt.labels))
        ]))
        
    def forward(self, x):
        x = self.body(x)
        x = x.view(x.size()[0], -1)
        x = self.tail(x)
        return x
