import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, Set2Set
from torch_geometric.nn.glob import GlobalAttention


class GraphAttentionLayer(torch.nn.Module):
    def __init__(self, dim, layer_num, heads=4):
        super(GraphAttentionLayer, self).__init__()
        self.layer_num = layer_num
        self.heads = heads

        # Use GATConv instead of GCNConv
        self.GraphAttention = nn.ModuleList(
            [GATConv(dim, dim // heads, heads=heads, concat=True) for _ in range(layer_num)]
        )

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.GraphAttention:
            x = F.relu(layer(x, edge_index, edge_weight))
        return x



class GraphAttentionNet(torch.nn.Module):
    def __init__(self, net_params):
        super(GraphAttentionNet, self).__init__()

        self.device = net_params['device']
        self.num_layers = net_params['num_layers']
        self.vocablen = net_params['vocablen']
        self.edgelen = net_params['edgelen']
        self.embedding_dim = net_params['embedding_dim']
        self.heads = net_params['heads']  # Number of attention heads, default to 1

        self.embed = nn.Embedding(self.vocablen, self.embedding_dim)
        self.edge_embed = nn.Embedding(20, self.embedding_dim)
        self.gat_layers = GraphAttentionLayer(self.embedding_dim, self.num_layers, heads=self.heads)
        self.mlp_gate = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Sigmoid())
        self.pool = GlobalAttention(gate_nn=self.mlp_gate)

        self.fc1 = torch.nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc2 = torch.nn.Linear(self.embedding_dim, 1)

    def forward(self, data):
        x1, x2, edge_index1, edge_index2, edge_attr1, edge_attr2 = data

        x1 = self.embed(x1).squeeze(1)
        x2 = self.embed(x2).squeeze(1)

        if edge_attr1 is None:
            edge_weight1 = None
            edge_weight2 = None
        else:
            edge_weight1=self.edge_embed(edge_attr1)
            edge_weight1=edge_weight1.squeeze(1)
            edge_weight1 = edge_weight1.max(dim=1)[0]
            edge_weight2=self.edge_embed(edge_attr2)
            edge_weight2=edge_weight2.squeeze(1)
            edge_weight2 = edge_weight2.max(dim=1)[0]

        x1 = self.gat_layers(x1, edge_index1, edge_weight1)
        x2 = self.gat_layers(x2, edge_index2, edge_weight2)

        batch1 = torch.zeros(x1.size(0), dtype=torch.long).to(self.device)
        batch2 = torch.zeros(x2.size(0), dtype=torch.long).to(self.device)

        x1 = self.pool(x1, batch=batch1)
        x2 = self.pool(x2, batch=batch2)

        concatenated_emb = torch.cat((x1, x2), dim=1)
        concatenated_emb = F.relu(self.fc1(concatenated_emb))
        out = self.fc2(concatenated_emb)

        pred = out.view(-1)

        return pred