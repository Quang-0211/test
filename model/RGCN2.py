import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCN2(nn.Module):
    """The PRDetect graph classifier with relation-aware convolutions."""

    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        output_dim=64,
        num_relations=128,
        num_bases=8,
        num_blocks=None,
        dropout=0.5,
    ):
        super().__init__()
        kwargs = {"num_bases": num_bases} if num_blocks is None else {"num_blocks": num_blocks}
        self.conv1 = RGCNConv(input_dim, hidden_dim, num_relations, **kwargs)
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations, **kwargs)
        self.fc = nn.Linear(output_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        if not hasattr(data, "edge_type"):
            raise ValueError("RGCN2 requires data.edge_type")
        x = self.conv1(data.x, data.edge_index, data.edge_type)
        x = self.dropout(F.relu(x))
        x = self.conv2(x, data.edge_index, data.edge_type)
        x = self.dropout(F.relu(x))
        return (self.fc(x).mean(dim=0, keepdim=True)).sigmoid()
