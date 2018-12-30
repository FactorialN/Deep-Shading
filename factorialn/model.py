import torch
from torch import nn
from torch.nn import functional as F
import loss

class Model(nn.Module):

    def __init__(self, n, channel, learning_rate=0.1, learning_rate_decay=0.995):
        super(Model,self).__init__()

        self.n=n
        self.channel=channel

        self.conv=[]
        self.convb=[]
        tmp=n
        for z in range(20):
            if tmp<=16:
                for y in range(z):
                    self.convb.append(nn.Conv2d(channel*(2**(y+(2 if y<z-1 else 1)))+channel*(2**y),channel*(2**(y+1)),3,padding=1))
                break
            self.conv.append(nn.Conv2d(channel*(2**z),channel*(2**(z+1)),3,padding=1))
            tmp/=2
        self.lin=nn.Linear(channel*2,1)


    def forward(self, x, o):
        x = torch.tensor(x,dtype=torch.float).permute(0,3,1,2)
        o = torch.tensor(o,dtype=torch.float)

        n = self.n
        u = self.channel
        x_i = x
        namei = 0
        sv = []
        while n > 16:
            sv.append(x_i)
            l_relu = F.leaky_relu(self.conv[namei](x_i))
            x_i = F.avg_pool2d(l_relu, 2)
            n /= 2
            u *= 2
            namei += 1
        x_i = F.leaky_relu(x_i)
        while n < self.n:
            namei -= 1
            us = F.upsample(x_i, scale_factor=2, mode='bilinear')
            cct = torch.cat([us, sv[namei]], dim=1)
            l_relu = F.leaky_relu(self.convb[namei](cct))
            n *= 2
            u /= 2
            x_i = l_relu

        x_i=x_i.permute(0,2,3,1)

        logits=self.lin(x_i)
        logits = (torch.tanh(logits) + 1) / 2

        return loss.CrossEntropyLoss(o, logits)
