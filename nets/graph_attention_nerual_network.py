import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.glob import GlobalAttention


class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads, device, dropout=0.0, concat=True):
        super(GATLayer, self).__init__(aggr='add')  # "Add" aggregation
        self.device = device
        self.heads = heads
        self.concat = concat  # Whether to concatenate or average head outputs
        self.out_channels = out_channels
        self.dropout = nn.Dropout(dropout)

        # Transformation for node features
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention mechanism
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        nn.init.xavier_uniform_(self.att)

        # Bias
        if concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        else:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        # Linear transformation of input features
        x = self.lin(x)
        x = x.view(-1, self.heads, self.out_channels)  # Shape: [N, heads, out_channels]

        # Propagate messages
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j, edge_index, size_i):
        # Compute attention coefficients
        alpha = torch.cat([x_i, x_j], dim=-1)  # Shape: [E, heads, 2 * out_channels]
        alpha = (alpha * self.att).sum(dim=-1)  # Shape: [E, heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Normalize attention coefficients
        alpha = softmax(alpha, edge_index[0], num_nodes=size_i)
        alpha = self.dropout(alpha)  # Apply dropout

        # Scale the messages
        return x_j * alpha.unsqueeze(-1)

    def update(self, aggr_out):
        # Aggregate outputs from multiple heads
        if self.concat:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        # Apply bias
        return aggr_out + self.bias


class GraphAttentionNet(nn.Module):
    def __init__(self, net_params):
        super(GraphAttentionNet, self).__init__()
        
        self.device = net_params['device']
        self.num_layers = net_params['num_layers']
        self.vocablen = net_params['vocablen']
        self.edgelen = net_params['edgelen']
        self.embedding_dim = net_params['embedding_dim']
        self.heads = net_params['heads']
        self.concat = net_params.get('concat', True)
        self.dropout = net_params.get('dropout', 0.0)
        
        # Embedding layers
        self.embed = nn.Embedding(self.vocablen, self.embedding_dim)
        self.edge_embed = nn.Embedding(self.edgelen, 1)  # Edge weights as scalars
        
        # GAT layers
        self.gat_layers = nn.ModuleList([
            GATLayer(
                in_channels=self.embedding_dim if i == 0 else self.heads * self.embedding_dim,
                out_channels=self.embedding_dim,
                heads=self.heads,
                concat=self.concat,
                dropout=self.dropout
            )
            for i in range(self.num_layers)
        ])
        
        # Global pooling
        self.mlp_gate = nn.Sequential(nn.Linear(self.heads * self.embedding_dim if self.concat else self.embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)