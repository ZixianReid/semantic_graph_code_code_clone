import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing,GatedGraphConv
from torch_geometric.utils import scatter
from torch_geometric.nn.glob import GlobalAttention
import inspect
getargspec = inspect.getfullargspec

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

class GMNlayer(MessagePassing):
    def __init__(self, in_channels, out_channels,device):
        super(GMNlayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.device=device
        self.out_channels = out_channels
        self.fmessage = nn.Linear(3*in_channels, out_channels)
        self.fnode = torch.nn.GRUCell(2*out_channels, out_channels, bias=True)
        self.__match_args__ = getargspec(self.match)[0][1:]
        self.__special_match_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__match_args__)
                                 if arg in special_args]
        self.__match_args__ = [
            arg for arg in self.__match_args__ if arg not in special_args
        ]


    def forward(self, x1,x2, edge_index1,edge_index2,edge_weight1,edge_weight2,mode='train'):


        m1=self.propagate(edge_index1,size=(x1.size(0), x1.size(0)), x=x1,edge_weight=edge_weight1)
        m2=self.propagate(edge_index2,size=(x2.size(0), x2.size(0)), x=x2,edge_weight=edge_weight2)

        scores = torch.mm(x1, x2.t())
        attn_1=F.softmax(scores,dim=1)

        attn_2=F.softmax(scores,dim=0).t()

        attnsum_1=torch.mm(attn_1,x2)
        attnsum_2=torch.mm(attn_2,x1)

        u1=x1-attnsum_1
        u2=x2-attnsum_2


        m1=torch.cat([m1,u1],dim=1)
        h1=self.fnode(m1,x1)
        m2=torch.cat([m2,u2],dim=1)
        h2=self.fnode(m2,x2)
        return h1,h2

    def message(self, x_i, x_j, edge_index,size,edge_weight=None):

        if type(edge_weight)==type(None):
            edge_weight=torch.ones(x_i.size(0),x_i.size(1)).to(self.device)
            m=F.relu(self.fmessage(torch.cat([x_i,x_j,edge_weight],dim=1)))
        else:
            m=F.relu(self.fmessage(torch.cat([x_i,x_j,edge_weight],dim=1)))
        return m

    def match(self, edge_index_i, x_i, x_j, size_i):
        return


    def update(self, aggr_out):

        return aggr_out



class GraphMatchNet(nn.Module):
    def __init__(self, net_params):
        super(GraphMatchNet, self).__init__()


        self.device=net_params['device']
        self.num_layers=net_params['num_layers']
        self.vocablen=net_params['vocablen']
        self.edgelen = net_params['edgelen']
        self.embedding_dim=net_params['embedding_dim']
        

        self.embed=nn.Embedding(self.vocablen,self.embedding_dim)
        self.edge_embed=nn.Embedding(self.edgelen,self.embedding_dim)
        self.gmnlayer=GMNlayer(self.embedding_dim,self.embedding_dim,self.device)
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
        for i in range(self.num_layers):
            x1, x2 = self.gmnlayer.forward(x1, x2, edge_index1, edge_index2, edge_weight1, edge_weight2, mode='train')

        batch1=torch.zeros(x1.size(0),dtype=torch.long).to(self.device)
        batch2=torch.zeros(x2.size(0),dtype=torch.long).to(self.device)
        hg1=self.pool(x1,batch=batch1)
        hg2=self.pool(x2,batch=batch2)

        return hg1,hg2
    



