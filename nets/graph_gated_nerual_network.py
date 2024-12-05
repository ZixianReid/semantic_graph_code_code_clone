import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GatedGraphConv
from torch_geometric.utils import degree, remove_self_loops, add_self_loops, softmax,scatter
#from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.glob import GlobalAttention


class GGNN(torch.nn.Module):
    def __init__(self,net_params):
        super(GGNN, self).__init__()

        self.device=net_params['device']
        self.num_layers=net_params['num_layers']
        self.vocablen=net_params['vocablen']
        self.edgelen = net_params['edgelen']
        self.embedding_dim=net_params['embedding_dim']


        #self.num_layers=num_layers
        self.embed=nn.Embedding(self.vocablen,self.embedding_dim)
        self.edge_embed=nn.Embedding(20,self.embedding_dim)
        #self.gmn=nn.ModuleList([GMNlayer(embedding_dim,embedding_dim) for i in range(num_layers)])
        self.ggnnlayer=GatedGraphConv(self.embedding_dim,self.num_layers)
        self.mlp_gate=nn.Sequential(nn.Linear(self.embedding_dim,1),nn.Sigmoid())
        self.pool=GlobalAttention(gate_nn=self.mlp_gate)

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


        x1 = self.ggnnlayer(x1, edge_index1)
        x2 = self.ggnnlayer(x2, edge_index2)

        batch1=torch.zeros(x1.size(0),dtype=torch.long).to(self.device)
        batch2=torch.zeros(x2.size(0),dtype=torch.long).to(self.device)
        hg1=self.pool(x1,batch=batch1)
        hg2=self.pool(x2,batch=batch2)

        return hg1, hg2

