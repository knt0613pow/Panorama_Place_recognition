import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.layer import *
from utils.maketable import *


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class PHD_VGG16(nn.Module):
    def __init__(self, conv_tables, adj_tables, pooling_tables, div):
        super(PHD_VGG16, self).__init__()
        self.conv_tables = conv_tables
        self.adj_tables = adj_tables
        self.pooling_tables = pooling_tables
        self.div = div

        self.layer1 = nn.Sequential(
            PHD_conv2d(in_dim = 3,  out_dim = 64, stride = 1, conv_table = conv_tables[div]),
            PHD_conv2d(in_dim = 64, out_dim = 64, stride = 1, conv_table = conv_tables[div]),
        )
        self.layer2 = nn.Sequential(
            PHD_conv2d(in_dim = 64, out_dim = 128, stride = 1, conv_table = conv_tables[div-1]),
            PHD_conv2d(in_dim = 128, out_dim = 128, stride = 1, conv_table = conv_tables[div-1]),
        )
        self.layer3 = nn.Sequential(
            PHD_conv2d(in_dim = 128, out_dim = 256, stride = 1, conv_table = conv_tables[div-2]),
            PHD_conv2d(in_dim = 256, out_dim = 256, stride = 1, conv_table = conv_tables[div-2]),
            PHD_conv2d(in_dim = 256, out_dim = 256, stride = 1, conv_table = conv_tables[div-2]),
        )

        self.layer4 = nn.Sequential(
            PHD_conv2d(in_dim = 256, out_dim = 512, stride = 1, conv_table = conv_tables[div-3]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-3]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-3]),
        )

        self.layer5 = nn.Sequential(
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-4]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-4]),
            PHD_conv2d(in_dim = 512, out_dim = 512, stride = 1, conv_table = conv_tables[div-4]),
        )

    def forward(self, x):
        l1 = self.layer1(x)
        #adj_table, pooling_table
        l1 = PHD_maxpool(l1, self.adj_tables[self.div], self.pooling_tables[self.div-1])
        l2 = self.layer2(l1)
        l2 = PHD_maxpool(l2, self.adj_tables[self.div-1], self.pooling_tables[self.div-2])
        l3= self.layer3(l2)
        l3 = PHD_maxpool(l3, self.adj_tables[self.div-2], self.pooling_tables[self.div-3])
        l4= self.layer4(l3)
        l4 = PHD_maxpool(l4, self.adj_tables[self.div-3], self.pooling_tables[self.div-4])
        l5= self.layer5(l4)
        l5 = PHD_maxpool(l5, self.adj_tables[self.div-4], self.pooling_tables[self.div-5])
        return l5



class PHD_delf_model(BaseModel):
    def __init__(self, subdivision = 5):
        super(PHD_delf_model, self).__init__()
        conv_tables = []
        adj_tables = []
        for i in range(0, subdivision+1):
            conv_tables.append(make_conv_table(i))
            adj_tables.append(make_adjacency_table(i))
        pooling_tables = make_pooling_table(subdivision)

        self.backbone = PHD_VGG16(conv_tables, adj_tables, pooling_tables, subdivision)
        self.att = PHD_SpatialAttention(512)
    
    def forward(self, x):
        output = self.backbone(x) # N, 512, 1, 20*4**(div-5)
        att_score = self.att(output) # N , 1, 1, 20*4**(div-5)
        return output* att_score


