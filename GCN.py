import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()

        # 定义GraphSAGE的层
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一个GraphSAGE卷积层，激活函数为ReLU
        x = F.relu(self.conv1(x, edge_index))
        # 第二个GraphSAGE卷积层，激活函数为ReLU
        x = F.relu(self.conv2(x, edge_index))
        return x


