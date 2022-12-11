import torch
import numpy as np
from torch.nn import init
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
from torch.nn import functional as F
from torch.nn import Module
import math

class PHD_conv2d_size1(Module):
    def __init__(self, 
                in_dim:int,
                out_dim:int,
                activation = 'ReLU',
                ) -> None:
        super(PHD_conv2d_size1, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = 1
        self.activation = getattr(torch.nn, activation)()
        shape = np.asarray([out_dim, in_dim, 1, 1])
        self.weight = Parameter(torch.empty(*shape),)
        self.bias = Parameter(torch.empty(shape[0]),) 
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        '''
        input : (N, C, H = 1, W = 20 * (4**division))
    
        '''
        output = F.conv2d(input,self.weight, self.bias, self.stride)
        output = self.activation(output)
        return output    

class PHD_conv2d(Module):
    def __init__(self, 
                in_dim:int,
                out_dim:int,
                stride : int,
                activation = 'ReLU',
                conv_table = None,
                ) -> None:
        super(PHD_conv2d, self).__init__()

        assert conv_table is not None

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.stride = stride
        self.activation = getattr(torch.nn, activation)()
        self.conv_table = conv_table  # (W, 10)
        shape = np.asarray([out_dim, in_dim, 10, 1])
        self.weight = Parameter(torch.empty(*shape),)
        self.bias = Parameter(torch.empty(shape[0]),) 

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        '''
        input : (N, C, H = 1, W = 20 * (4**division))
        '''
        x = input[..., self.conv_table.T].squeeze()
        output = F.conv2d(x,self.weight, self.bias, self.stride)
        output = self.activation(output)
        return output
    
def PHD_maxpool(input, adj_table, pooling_table):
    '''
    input (N, C, H =1, W = 20*(4 ** division))
    adj_table (W, 4)
    pooling_table (20*(4**(division-1)))
'''
    output = input[..., adj_table.T].squeeze() # (N, C, 4, W,)
    output = output[..., pooling_table] # (N, C, 4, W/4, )
    output = torch.max(output, dim=2,keepdim=True)[0] # (N, C/ 1, W/4)
    return output

def PHD_avgpool(input):
    '''
    input (N, C, H =1, W = 20*(4 ** division))
    '''
    output = torch.mean(input, (2, 3)) # (N, C)
    return output    

class PHD_SpatialAttention(Module):
    def __init__(self, in_c):
        super(PHD_SpatialAttention, self).__init__()
        self.conv1  = PHD_conv2d_size1(in_c, 1024)
        self.conv2  = PHD_conv2d_size1(1024, 1, activation = 'Softplus')

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)

        return output
