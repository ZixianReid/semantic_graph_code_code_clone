
from train.train_gmn import train_gmn




def trainer(MODEL_NAME, dataset, params, net_params, dirs):
    trainers = {
        'graph_match_nerual_network': train_gmn,
    }
    trainers[MODEL_NAME](MODEL_NAME, dataset, params, net_params, dirs)

