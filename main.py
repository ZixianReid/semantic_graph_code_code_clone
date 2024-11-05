import argparse
import json
import os
import time
from data.data import LoadData
from util.setting import init_logging, log
from data.dataset_builder import Dataset
from nets.load_net import gnn_model
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter


def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):

    #dataset loading
    vocablen, trainset, valset, testset = dataset.vocab_length, dataset.train_data, dataset.val_data, dataset.test_data


    device = net_params['device']
    net_params['vocablen'] = vocablen
    log.info(f"Vocab length: {vocablen}")
    log.info(f"Trainset length: {len(trainset)}")
    log.info(f"Valset length: {len(valset)}")
    log.info(f"Testset length: {len(testset)}")


    # model setting
    log.info("Model setting")
    model = gnn_model(MODEL_NAME, net_params)
    model.to(device)

    #optmizer and scheduler setting
    log.info("Optimizer and scheduler setting")
    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)


    # data loader
    log.info("Data loader setting")
    train_loader = DataLoader(trainset, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)

    from train.train import train_epoch, evaluate_network

    # start to train
    log.info("Start to train")
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_pre, epoch_train_recall, epoch_train_f1, optimizer = train_epoch(model, optimizer, device, train_loader,
                                                                           epoch)
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')



def main():
    parser = argparse.ArgumentParser(description='code clone detection')
    parser.add_argument('-l', '--logging', type=int, default=10,
                        help='Log level [10-50] (default: 10 - Debug)')
    parser.add_argument('--config')
    parser.add_argument('--language')
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--batch_size')
    parser.add_argument('--report', type=str, default="",
                        help='file name to report training logs.') 

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    
        
    params = config['params']
    net_params = config['net_params']

    # initial logger
    init_logging(args.logging, args.report)

    # dataset
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    
    dataset = LoadData(DATASET_NAME)
    # model
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']


    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(
        config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file


    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')

    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')
    
    
    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)


main()