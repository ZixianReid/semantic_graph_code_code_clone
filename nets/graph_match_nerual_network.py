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
    def propagate_match(self, edge_index, size=None, **kwargs):
        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        match_args = []
        #print(self.__special_match_args__)
        #print(self.__match_args__)
        #print(ij.keys())
        for arg in self.__match_args__:
            #print(arg)
            #print(arg[-2:])
            if arg[-2:] in ij.keys():
                tmp = kwargs.get(arg[:-2], None)
                if tmp is None:  # pragma: no cover
                    match_args.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if tmp[1 - idx] is not None:
                            if size[1 - idx] is None:
                                size[1 - idx] = tmp[1 - idx].size(0)
                            if size[1 - idx] != tmp[1 - idx].size(0):
                                raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    match_args.append(tmp)
                #print(tmp)
            else:
                match_args.append(kwargs.get(arg, None))

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_match_args__:
            if arg[-2:] in ij.keys():
                match_args.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                match_args.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]
        #print(match_args)
        out_attn = self.match(*match_args)
        #print(out_attn.size())
        out_attn = scatter(out_attn, edge_index[i], dim_size=size[i], reduce=self.aggr,)
        #print(out_attn.size())
        out_attn = self.update(out_attn, *update_args)
        #out=torch.cat([out,out_attn],dim=1)
        #print(out.size())
        return out_attn




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
    



