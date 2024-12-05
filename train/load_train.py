
from train.train_gmn import train_gmn
from train.train_gcn import train_gcn



def trainer(MODEL_NAME, dataset, params, net_params, dirs):
    trainers = {
        'graph_match_nerual_network': train_gmn,
        'graph_convolution_nerual_network': train_gcn,
        'gated_graph_nerual_network': train_gmn,
    }
    trainers[MODEL_NAME](MODEL_NAME, dataset, params, net_params, dirs)

