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

from tensorboardX import SummaryWriter

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
    parser.add_argument('--config', default='/home/zixian/PycharmProjects/semantic_graph_code_code_clone/configs/semantic_code_graph_gmm_bcb.json')
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
    

    
    trainer(MODEL_NAME, dataset, params, net_params, dirs)

main()