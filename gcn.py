import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)  # First graph convolutional layer
        self.conv2 = GCNConv(16, num_classes)  # Second graph convolutional layer for classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Second layer
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)