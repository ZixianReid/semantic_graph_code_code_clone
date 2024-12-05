from nets.graph_match_nerual_network import GraphMatchNet
from nets.graph_convolution_nerual_network import GraphConvNet
from nets.graph_gated_nerual_network import GGNN

def GraphMatchNeuralNetwork(net_params):
    return GraphMatchNet(net_params)

def GraphConvolutionNerualNetwork(net_params):
    return GraphConvNet(net_params)

def GatedGraphNerualNetwork(net_params):
    return GGNN(net_params)

def gnn_model(MODEL_NAME, net_pararms):
    models = {
        'graph_match_nerual_network': GraphMatchNeuralNetwork,
        'graph_convolution_nerual_network': GraphConvolutionNerualNetwork,
        'gated_graph_nerual_network': GatedGraphNerualNetwork
    }

    return models[MODEL_NAME](net_pararms)


