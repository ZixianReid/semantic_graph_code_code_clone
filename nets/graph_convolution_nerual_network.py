import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GatedGraphConv, GCNConv, Set2Set
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.glob import GlobalAttention



# class GCNLayer(MessagePassing):
#     def __init__(self, in_channels, out_channels, device):
#         super(GCNLayer, self).__init__(aggr='add')  # "Add" aggregation
#         self.device = device
#         self.lin = nn.Linear(in_channels, out_channels)
#         self.root_emb = nn.Parameter(torch.Tensor(in_channels, out_channels))
#         nn.init.xavier_uniform_(self.root_emb)

#     def forward(self, x, edge_index, edge_weight=None):
#         # Add self-loops to the adjacency matrix
#         edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=x.size(0))

#         # Compute normalization (symmetric normalization)
#         row, col = edge_index
#         deg = degree(row, x.size(0), dtype=x.dtype)
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
#         norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

#         return self.propagate(edge_index, x=x, norm=norm)

#     def message(self, x_j, norm):
#         # Apply normalization to the messages
#         return torch.matmul(norm, x_j)

#     def update(self, aggr_out, x):
#         # Combine the aggregated messages with the original node features
#         return aggr_out + torch.matmul(x, self.root_emb)


# class GraphConvNet(nn.Module):
#     def __init__(self, net_params):
#         super(GraphConvNet, self).__init__()
#         self.device = net_params['device']
#         self.num_layers = net_params['num_layers']
#         self.vocablen = net_params['vocablen']
#         self.edgelen = net_params['edgelen']
#         self.embedding_dim = net_params['embedding_dim']

#         self.embed = nn.Embedding(self.vocablen, self.embedding_dim)

#         self.edge_embed = nn.Embedding(self.edgelen, 1) 

#         self.gcn_layers = nn.ModuleList([
#             GCNLayer(self.embedding_dim, self.embedding_dim, self.device)
#             for _ in range(self.num_layers)
#         ])
#         self.mlp_gate = nn.Sequential(nn.Linear(self.embedding_dim, 1), nn.Sigmoid())

#         self.pool = GlobalAttention(gate_nn=self.mlp_gate)

    
#     def forward(self, data):
#         x1,x2, edge_index1, edge_index2,edge_attr1,edge_attr2 = data
#         x1 = self.embed(x1)
#         x1 = x1.squeeze(1)
#         x2 = self.embed(x2)
#         x2 = x2.squeeze(1)

#         if type(edge_attr1) == type(None):
#             edge_weight1 = None
#             edge_weight2 = None
#         else:
#             edge_weight1 = self.edge_embed(edge_attr1)
#             edge_weight1 = edge_weight1.squeeze(1)
#             edge_weight2 = self.edge_embed(edge_attr2)
#             edge_weight2 = edge_weight2.squeeze(1)
        
#         for gcn_layer in self.gcn_layers:
#             x1 = gcn_layer(x1, edge_index1, edge_weight1)
#             x2 = gcn_layer(x2, edge_index2, edge_weight2)
        

#         batch1=torch.zeros(x1.size(0),dtype=torch.long).to(self.device)
#         batch2=torch.zeros(x2.size(0),dtype=torch.long).to(self.device)

#         hg1 = self.pool(x1, batch=batch1)
#         hg2 = self.pool(x2, batch=batch2)

#         return hg1, hg2



class GraphConvLayer(torch.nn.Module):
    def __init__(self, dim, layer_num):
        super(GraphConvLayer, self).__init__()

        self.GraphConv = GCNConv(dim, dim)
        self.layer_num = layer_num

        
    def forward(self, x, edge_index):
        for i in range(self.layer_num):
            m = F.relu(self.GraphConv(x, edge_index))
        return m

class GraphConvNet(torch.nn.Module):
    def __init__(self,net_params):
        super(GraphConvNet, self).__init__()

        self.device=net_params['device']
        self.num_layers=net_params['num_layers']
        self.vocablen=net_params['vocablen']
        self.edgelen = net_params['edgelen']
        self.embedding_dim=net_params['embedding_dim']


        #self.num_layers=num_layers
        self.embed=nn.Embedding(self.vocablen,self.embedding_dim)
        self.edge_embed=nn.Embedding(20,self.embedding_dim)
        self.gcn_layers = GraphConvLayer(self.embedding_dim, self.num_layers)
        self.mlp_gate=nn.Sequential(nn.Linear(self.embedding_dim,1),nn.Sigmoid())
        self.set2set = Set2Set(self.embedding_dim, processing_steps=3)
        self.fc1 =  torch.nn.Linear(self.embedding_dim * 4, self.embedding_dim)

        self.fc2 = torch.nn.Linear(self.embedding_dim, 1)
        

    def forward(self, data):
        x1,x2, edge_index1, edge_index2,edge_attr1,edge_attr2 = data
        x1 = self.embed(x1)
        x1 = x1.squeeze(1)
        x2 = self.embed(x2)
        x2 = x2.squeeze(1)

        if type(edge_attr1)==type(None):
            edge_weight1=None
            edge_weight2=None
        else:
            edge_weight1=self.edge_embed(edge_attr1)
            edge_weight1=edge_weight1.squeeze(1)
            edge_weight2=self.edge_embed(edge_attr2)
            edge_weight2=edge_weight2.squeeze(1)

        x1 = self.gcn_layers(x1, edge_index1)

        x2 = self.gcn_layers(x2, edge_index2)

        x1 = self.set2set(x1)

        x2 = self.set2set(x2)
        
        concatenatedEmb = torch.cat((x1,x2),dim=1)

        concatenatedEmb = F.relu(self.fc1(concatenatedEmb))

        out = self.fc2(concatenatedEmb)

        pred = out.view(-1)

        return pred


