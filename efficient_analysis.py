import argparse
import json
import os
import time
import torch
import numpy as np
from data.data import LoadData
from util.setting import init_logging, log, view_params
from data.dataset_builder import Dataset
from nets.load_net import gnn_model
from train.load_train import trainer
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import argparse
import json
import os
import time
import torch
import numpy as np
from data.data import LoadData
from util.setting import init_logging, log, view_params
from data.dataset_builder import Dataset
from nets.load_net import gnn_model
from train.load_train import trainer
from train.train_gmn import load_and_evaluate_model
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from tensorboardX import SummaryWriter



def get_size(obj):
    """ Recursively calculates the size of objects in bytes """
    if isinstance(obj, dict):
        return sum(get_size(v) for v in obj.values()) + sys.getsizeof(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return sum(get_size(v) for v in obj) + sys.getsizeof(obj)
    elif isinstance(obj, np.ndarray) or isinstance(obj, torch.Tensor):
        return obj.element_size() * obj.nelement()
    else:
        return sys.getsizeof(obj)

def compute_total_graph_dict_size(graph_dict):
    """
    Compute total memory footprint of the graph_dict, including all graphs stored within it.

    :param graph_dict: Dictionary storing graphs in the format {filename: [[x, edge_index, edge_attr], astlength]}
    :return: Total size in bytes
    """
    total_size = get_size(graph_dict)
    return total_size

def compute_average_graph_density(graph_dict):
    """
    Computes the average graph density across all graphs stored in graph_dict.

    :param graph_dict: Dictionary storing graphs in the format {filename: [[x, edge_index, edge_attr], astlength]}
    :return: Average Graph Density
    """
    total_density = 0
    graph_count = len(graph_dict)

    if graph_count == 0:
        return 0  # No graphs to compute

    for file_name, graph_data in graph_dict.items():
        x, edge_index, edge_attr = graph_data[0]  # Extract graph components
        num_nodes = len(x)  # Number of nodes |V|
        num_edges = len(edge_index[0])  # Number of edges |E|

        # Compute density for this graph
        max_possible_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0

        total_density += density

    # Compute average density
    avg_density = total_density / graph_count
    return avg_density

def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device



def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param

def create_dirs(dirs):
    for dir_ in dirs:
        if not os.path.exists(dir_):
            os.makedirs(dir_)

def main():
    parser = argparse.ArgumentParser(description='code clone detection')
    parser.add_argument('-l', '--logging', type=int, default=20,
                        help='Log level [10-50] (default: 10 - Debug)')
    parser.add_argument('--config', default='/home/zixian/PycharmProjects/semantic_graph_code_code_clone/configs/benchmark_without_value_GMN/BCB/AST_CFG_DFG_FA_GMN_BCB.json')
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--language')
    parser.add_argument('--out_dir')
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--seed')
    parser.add_argument('--epochs')
    parser.add_argument('--init_lr')
    parser.add_argument('--batch_size')
    parser.add_argument('--lr_reduce_factor')
    parser.add_argument('--lr_schedule_patience')
    parser.add_argument('--min_lr')
    parser.add_argument('--weight_decay')
    parser.add_argument('--print_epoch_interval')
    parser.add_argument('--max_time')
    parser.add_argument('--report', type=str, default="clone_bcb_2",
                        help='file name to report training logs.') 

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    # out_dir
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir'] 


    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset_params']['name']
    
    LANGUAGE = config['language']

    # model
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    #dirs
    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + LANGUAGE + DATASET_NAME +  "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" +  LANGUAGE + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + LANGUAGE + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + LANGUAGE + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    create_dirs(dirs)

    # initial logger
    init_logging(args.logging, args.report, root_log_dir)

        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
  
        

    # paramters setting
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)


    #network parameters setting
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']


    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')


    dataset_params = config['dataset_params']

    # view parameters
    view_params(params, dataset_params, net_params)

    # dataset
    dataset = LoadData(LANGUAGE, dataset_params)

    graph_dict = dataset.graph_dict




    total_memory_size = compute_total_graph_dict_size(graph_dict)
    print(f"Total Memory Used by graph_dict: {total_memory_size / (1024 * 1024):.2f} MB")
    average_density = compute_average_graph_density(graph_dict)
    print(f"Average Graph Density: {average_density:.4f}")


    model_path = '/home/zixian/PycharmProjects/semantic_graph_code_code_clone/run/out_gmn/BCB/AST_GMN_BCB/checkpoints/graph_match_nerual_network_JavaBigCloneBench_GPU0_14h53m56s_on_Jan_06_2025/model_19.pth'

    net_params['vocablen'] = dataset.vocab_length
    load_and_evaluate_model(model_path, dataset.test_data, params, net_params)




main()