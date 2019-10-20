import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class CNN_Text(nn.Module):
    
    def __init__(self, input_channel, kernel_num, kernel_size, output, dropout, concat):
        super(CNN_Text, self).__init__()
        self.input_channel = input_channel
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.output = output


        if concat == 2:
            self.convs1 = nn.ModuleList([nn.Conv2d(self.input_channel, self.kernel_num, (K, 768*2)) for K in self.kernel_size])
        else:
            self.convs1 = nn.ModuleList([nn.Conv2d(self.input_channel, self.kernel_num, (K, 768)) for K in self.kernel_size])

        self.dropout = nn.Dropout(self.dropout)
        
        self.fc1 = nn.Linear(len(self.kernel_size)*self.kernel_num, self.output)


    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D) batch*1*128*768

        x = [F.relu(conv(x)).squeeze(-1) for conv in self.convs1] 

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1) # 横着拼接矩阵

        x = self.dropout(x)
        logit = self.fc1(x)
        
        return logit