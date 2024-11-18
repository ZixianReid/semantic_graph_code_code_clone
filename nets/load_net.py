from nets.graph_match_nerual_network import GraphMatchNet

def GraphMatchNeuralNetwork(net_params):
    return GraphMatchNet(net_params)


def gnn_model(MODEL_NAME, net_pararms):
    models = {
        'graph_match_nerual_network': GraphMatchNeuralNetwork,
    }

    return models[MODEL_NAME](net_pararms)


